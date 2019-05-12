#==============================================================================
# Point neuron (I&F model). The model simulates AMPA, NMDA and
# GABA mechanisms. This script is used as a simple neuronal model for
# comparison purposes
#==============================================================================

from brian import *
from brian.library.ionic_currents import *
from brian.library.IF import *
import numpy as np
import time
import math
from brian.library.electrophysiology import *

reinit(states = True)
clear(erase = True, all = True)
def synchronous_input(A, N_pre):
    C = []
    for jj in range(len(A)):
        for ll in xrange(N_pre):
            B = list(A[jj])
            B[0] = ll
            B = tuple(B)
            C.append(B)
    C.sort(key = lambda tup: tup[1])
    return C

I_injected = range(-100, 210, 20)
I_injected = [0]
Spikes = np.zeros(len(I_injected))
Vpeak  = np.zeros(len(I_injected))

for i in range(len(I_injected)):

#    print
#    print "# of synapses: " + str(i)
#    print
    reinit(states = True)
    clear(erase = True, all = True)
    defaultclock.dt = 0.1 * ms
    # Number of presynaptic neurons
    N_pre  = 1
    N_basket = 1
    # Parameters
    gl_b         =  18.054  * nS # leakage conductance
    El_b         = -52      * mV # reversal-resting potential
    Cm_b         =   0.1793 * nF # membrane capacitance
    v_th_b       = -39      * mV # threshold potential
    v_reset_b    = -45      * mV # reset potential
    DeltaT_b     =   2      * mV # slope factor

    # Morphology
    # Soma
#    length_basket   =   (20 + 3500) * um
#    diam_basket     =   0.9 * um
#    area_basket     = math.pi * diam_basket * length_basket

    # Synaptic Kinetics
    E_nmda       =   0      * mV  # NMDA reversal potential
    E_ampa       =   0      * mV  # AMPA reversal potential
    E_gaba       =  -70     * mV  # GABA reversal potential

    g_ampa       =  0.38 * nS     # AMPA maximum conductance
    g_nmda       =  1.3  * nS # NMDA maximum conductance
    g_gaba       =  1.6  * nS  # GABA maximum conductance

    t_nmda_decay = 130      * ms  # NMDA decay time constant
    t_nmda_rise  =  10      * ms  # NMDA rise time constant
    t_ampa_decay =   4.2    * ms  # AMPA decay time constant
    t_ampa_rise  =   1.2    * ms  # AMPA rise time constant
    t_gaba_decay =   5.5    * ms  # GABA decay time constant
    t_gaba_rise  =   0.26   * ms  # GABA rise time constant

    gama         =   0.072  * mV**-1
    alpha_nmda   =   0.5    * ms**-1
    alpha_ampa   =   1      * ms**-1
    alpha_gaba   =   1      * ms**-1

    # Synaptic current equations
    eq_soma_b = Equations('''
    I_syn_b  = I_nmda + I_ampa + I_gaba - I_inj - clamp*i_inj         : amp
    I_nmda = g_nmda*(vm - E_nmda)*s_nmda*1./(1 + exp(-gama*vm)/3.57)  : amp
    I_ampa = g_ampa*(vm - E_ampa)*s_ampa                              : amp
    I_gaba = g_gaba*(vm - E_gaba)*s_gaba                              : amp
    I_inj                                                             : amp
    s_nmda                                                            : 1
    s_ampa                                                            : 1
    s_gaba                                                            : 1
    clamp                                                             : 1
    ''')

    # Brette-Gerstner
    basket_eqs = Brette_Gerstner(Cm_b, gl_b, El_b, v_th_b, DeltaT, tauw = 100 * ms, a = .1 * nS)
    basket_eqs += IonicCurrent('I = I_syn_b : amp')
    basket_eqs += eq_soma_b
    basket_eqs += voltage_clamp(v_cmd = -70*mV)

    basket = NeuronGroup(N_basket, model = basket_eqs, threshold = 'vm > v_th_b',
                         reset = AdaptiveReset(Vr=v_reset_b, b = 0.0205*nA),
                         refractory = 2 * ms, compile = True)

    # Initialization of membrane potential
    basket.vm = El_b
    basket.clamp = 1
    #==========================================================================

    # Input stochastic for each run
    P0 = PoissonGroup(N_pre)
    tspike = 500 * ms
    spiketimes = [(0, tspike)]
    time_int = 0.1 * ms
#    for dummy_ii in xrange(20):
#        spiketimes.append((0, 500*ms + time_int))
#        time_int += 0.1 * ms
    spiketimes = synchronous_input(spiketimes, N_pre)
#    spiketimes = [(ii, tspike) for ii in xrange(N_pre)]
    P0 = SpikeGeneratorGroup(N_pre, spiketimes)
    #==========================================================================

    #==========================================================================
    # The NMDA/AMPA synapses

    # Synapses at 1st branch
    nmda_eqs = '''
    dj/dt = -j / t_nmda_decay + alpha_nmda * x * (1 - j) : 1
    dx/dt = -x / t_nmda_rise                             : 1
    wNMDA                                                : 1
    '''

    syn_NMDA = Synapses(P0, basket, model = nmda_eqs, pre = 'x += wNMDA', implicit=True, freeze=True)
    basket.s_nmda = syn_NMDA.j
    syn_NMDA[:, :]        = True
    syn_NMDA.wNMDA[:, :]  = 1.0
    syn_NMDA.delay[:, :]  = 3.3 * ms

    ampa_eqs = '''
    dy/dt = -y / t_ampa_decay + alpha_ampa * h * (1 - y) : 1
    dh/dt = -h / t_ampa_rise                             : 1
    wAMPA                                                : 1
    '''

    syn_AMPA = Synapses(P0, basket, model = ampa_eqs, pre = 'h += wAMPA', implicit=True, freeze=True)
    basket.s_ampa = syn_AMPA.y
    syn_AMPA[:, :]        = True
    syn_AMPA.wAMPA[:, :]  = 1.0
    syn_AMPA.delay[:, :]  = 3.3 * ms
    #==========================================================================


    #==========================================================================
    # # Monitors
    pre0 = SpikeMonitor(P0)
    S    = SpikeMonitor(basket)

    M0  = StateMonitor(basket, 'vm', record = True)


    M1  = StateMonitor(syn_NMDA, 'j', record = True)
    M2  = StateMonitor(syn_NMDA, 'x', record = True)
    M3  = StateMonitor(syn_AMPA, 'y', record = True)
    M4  = StateMonitor(syn_AMPA, 'y', record = True)

    M5  = StateMonitor(basket, 'I_gaba', record = True)
    M6  = StateMonitor(basket, 'I_ampa', record = True)
    M7  = StateMonitor(basket, 'I_nmda', record = True)
    M8  = StateMonitor(basket, 'I_syn_b', record = True)
    M9  = StateMonitor(basket, 'i_inj', record = True)
    M10  = StateMonitor(basket, 'w', record = True)
    #======================================================================
    #======================================================================
    # SIMULATION
    #======================================================================

    #Simulation run
    simt  = 200 * ms
    total_simt = 5*simt
    rateA =  0 * Hz
    rateB =  40 * Hz
    t_stim_init = 300
    t_stim_end  = 1300

#    @network_operation
#    def apply_soma_injections():
#        if (defaultclock.t>t_stim_init * ms) and (defaultclock.t<=t_stim_end*ms):
#            basket.I_inj = I_injected[i] * pA
#        else:
#            basket.I_inj = 0 * pA
#
#    @network_operation
#    def apply_poisson_rates():
#        if (defaultclock.t>t_stim_init * ms) and (defaultclock.t<=t_stim_end*ms):
#            P0.rate = rateB
#        else:
#            P0.rate = rateA

    print "Simulation running..."
    start_time = time.time()
    run(total_simt)
    duration = time.time() - start_time

    print "Number of somatic spikes: " + str(S.nspikes)
    print
    print "Simulation time: ", duration, "seconds"

#    if I_injected[i] < 0:
#        print "Input Resistance: " + str((M0[0][22000] * volt - El_b)/(I_injected[i]*pA))
#        print
#        print "Sag ratio: " + str((M0[0][22000]*volt - El_b)/(min(M0[0])*volt - El_b))
#    else:
#        print "Slow AHP: " + str((min(M0[0]) * volt - El_b))
#        print
#
#
#    #peak voltage
#    if I_injected[i] > 0:
#        Vpeak[i] = 1000*(max(M0[0])  - El_b / volt) # in mvolts
#    else:
#        Vpeak[i] = 1000*(min(M0[0])  - El_b/ volt) # in mvolts
    print "EPSC: " + str(min(M6[0]) * amp)    # for AMPA
    print "EPSC: " + str(min(M6[0] + min(M7[0])) * amp)    # for NMDA + AMPA
##=============================================================================
# Visualization

doVisualise = 0
step = 500
if doVisualise == 1:
    fig = figure(1)
    subplot(211)
    raster_plot(pre0)
    ylabel('# of input cells')
    xticks(range(0, int(total_simt/msecond) + step, step))
    subplot(212)
    raster_plot(S)
    ylabel('# of basket cells')
    xticks(range(0, int(total_simt/msecond) + step, step))
#    savefig('raster_plots(2).png', bbox_inches='tight', dpi = 1080)

    fig = figure(2)
    vm_soma = M0[0]
    for _,t in S.spikes:
        i = int(t / defaultclock.dt)
        vm_soma[i] = (El_b + 78*mV)
    plot(M0.times / ms, vm_soma / mV)
    ylabel('V_soma (mV)')
    xlabel('time (ms)')
#    savefig('basket_Iinj.png', bbox_inches='tight', dpi = 1080)

    fig = figure(3)
    for i in range(len(P0)):
        plot(M1.times / ms, M1[i])
    ylabel('NMDA')
    xlabel('time (ms)')

    fig = figure(4)
    for i in range(len(P0)):
        plot(M2.times / ms, M2[i])
    ylabel('x NMDA')
    xlabel('time (ms)')

    fig = figure(5)
    for i in range(len(P0)):
        plot(M3.times / ms, M3[i])
    ylabel('AMPA')
    xlabel('time (ms)')

    fig = figure(6)
    for i in range(size(basket)):
        plot(M5.times / ms, M5[i] / pA)
    ylabel('I_gaba (pA)')
    xlabel('time (ms)')


    fig = figure(7)
    for i in range(size(basket)):
        plot(M6.times / ms, M6[i] / pA)
    ylabel('I_ampa (pA)')
    xlabel('time (ms)')

    fig = figure(8)
    for i in range(size(basket)):
        plot(M7.times / ms, M7[i] / pA)
    ylabel('I_nmda (pA)')
    xlabel('time (ms)')

    fig = figure(9)
    for i in range(size(basket)):
        plot(M8.times / ms, M8[i] / pA)
    ylabel('I_syn (pA)')
    xlabel('time (ms)')

    fig = figure(10)
    for i in range(size(basket)):
        plot(M9.times / ms, M9[i] / pA)
    ylabel('i_inj (pA)')
    xlabel('time (ms)')

    show()

doData = 0
if doData == 1:
    fig = figure(1)
    plot(I_injected, Vpeak, '.')
    xlabel('Iinj (pA)')
    ylabel('Peak voltage from rest (mV)')

    ax = gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_label_text('I (pA)')
    ax.xaxis.set_label_coords(1, .26)
    ax.yaxis.set_label_text('V (mV)')
    ax.yaxis.set_label_coords(0.22, 0.8)
    ax.yaxis.get_label().set_rotation('horizontal')
#    savefig('BasketCell.png', bbox_inches = 'tight', dpi = 1080)
##==============================================================================