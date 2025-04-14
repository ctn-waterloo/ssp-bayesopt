import numpy as np

from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg

from lava.utils import weightutils as wu
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from lava.proc.graded.process import GradedReluVec, GradedVec

from lava.proc import embedded_io as eio
from lava.proc.io import sink, source

def to_fixed_pt(xs, exp):
    return (xs*2**exp).astype(np.int32)

def from_fixed_pt(xs, exp):
    return (xs*2**(-exp)).astype(float)

class ZeroPointQuantizer:
    def __init__(self, W):
        self.w_range = W.max() - W.min()
        self.w_range = 1 if self.w_range == 0 else self.w_range
        self.scale = 255/self.w_range
        self.zeropoint = (-self.scale * W.min() - 128).round()

    def quantize(self, W):
        return np.clip((W*self.scale + self.zeropoint).round(), -128,127).astype(np.int32)

    def dequantize(self, W_quant):
        return (W_quant - self.zeropoint)/self.scale


           

def make_dense(W,name='',exp=6,num_message_bits=24):

    fixed_pt_weights = to_fixed_pt(W,exp)
    return Dense(
                weights=fixed_pt_weights,
                weight_exp=-exp, 
                name=name+'_weights',
                num_message_bits=num_message_bits,
                )

def convert_synapse(tau, dt):
    return np.array([-np.expm1(-dt/tau)])
    #return np.array([np.exp(-dt/tau)])

def extract_params(model, sim):

    retval = {
            'W_inp': sim.data[model.all_ensembles[0]].scaled_encoders,
            'b_inp': sim.data[model.all_ensembles[0]].bias,
            }

    if hasattr(model.all_ensembles[0].neuron_type, 'tau_rc'): 
        tau_rc = model.all_ensembles[0].neuron_type.tau_rc
        retval['tau_rc'] = convert_synapse(tau_rc,sim.dt)
    for conn in model.all_connections:
        if conn.label == 'inp_conn':
            pass
        if conn.label == 'fb_conn':
            retval['W_fb'] = sim.data[conn].weights
            retval['tau_fb'] = convert_synapse(conn.synapse.tau,sim.dt)
        if conn.label == 'soln_conn':
            retval['W_dec'] = sim.data[conn].weights
        ###
        pass
    return retval

class LavaBoModel:
    def __init__(self, inp_exp=12, w_exp=6):
        self.inp_exp = inp_exp
        self.w_exp = w_exp

    def convert(self, params, init_guess, num_steps, num_pres, use_relu=False):
        '''
        Converts a recurrent BO circuit.

        params : A dictionary of:
            W_enc: encoder weights
            b_enc: encoder biases
            W_fb: weights on feedback connection
            W_dec: decoder weights
        '''
        ### set up the input
        self.num_steps = num_steps
        self.num_pres = num_pres

        inp_values = np.zeros((self.num_steps, init_guess.size+1)).astype(np.int32)
        inp_values[:self.num_pres,:-1] = np.tile(
                to_fixed_pt(init_guess, self.inp_exp),
                (self.num_pres,1),
                )
        inp_values[:,-1] = 1 ## added to so the bias will always be active.
        inp_values = inp_values.T
        self.ring_buffer = source.RingBuffer(data=inp_values)
        self.inport_adapter = eio.spike.PyToNxAdapter(
                            shape=(inp_values.shape[0],),
                            num_message_bits=24)

        ### Create the circuit
        ### add biases to input_weights
        W_inp = np.hstack((params['W_inp'],params['b_inp'][:,None]))
        W_fb = params['W_inp'] @ params['W_fb']

        print('Inp: ', np.log(255/(W_inp.max() - W_inp.min())),W_inp.max(), W_inp.min())
        print('FB: ', np.log(255/(W_fb.max() - W_fb.min())), W_fb.max(), W_fb.min())

        


        ### x[t] = W_enc@inp[t] + W_enc@W_fb@x[t-1]
        self.input_dw = make_dense(
                W_inp,
                exp=self.w_exp,
                name='enc_weights',
                num_message_bits=24,
                )
        if use_relu:
            self.soln_ns = GradedReluVec(
                shape=(W_inp.shape[0],),
                vth=0,
                ) 
        else:
            self.soln_ns = LIF(
                    shape=(W_inp.shape[0],),
                    vth=0,
                    du=to_fixed_pt(params['tau_fb'], 12),
                    dv=to_fixed_pt(params['tau_rc'], 12),
                    )
        

        self.fb_dense = make_dense(W_fb,
                    exp=self.w_exp,
                    num_message_bits=24 if use_relu else 0,
                    name='fb_weights',
                )
        self.W_dec = params['W_dec']

        ## make the output nodes

        self.out_adapter = eio.spike.NxToPyAdapter(
                shape=(W_inp.shape[0],), 
                num_message_bits=24 if use_relu else 0,
            )
        self.logger = sink.RingBuffer(
                shape=(W_inp.shape[0],), 
                buffer=num_steps,
            )
        ## connect the network
        self.ring_buffer.s_out.connect(self.inport_adapter.inp) 
        self.inport_adapter.out.connect(self.input_dw.s_in)
        self.input_dw.a_out.connect(self.soln_ns.a_in)
        self.soln_ns.s_out.connect(self.fb_dense.s_in)
        self.fb_dense.a_out.connect(self.soln_ns.a_in)
        self.soln_ns.s_out.connect(self.out_adapter.inp)
        self.out_adapter.out.connect(self.logger.a_in)

    def run(self):
        run_cfg = Loihi2HwCfg()
        print("Running on Loihi 2")
        retval = None
        try:
            self.soln_ns.run(condition=RunSteps(num_steps=self.num_steps),
                            run_cfg=run_cfg)
            out_graded_spike_data = self.logger.data.get()
            floating_out_data = from_fixed_pt(
                    out_graded_spike_data, 
                    self.inp_exp,
                )
            retval = self.W_dec @ floating_out_data
        finally:
            self.soln_ns.stop()
        return retval
