import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time
import os

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')
layers_with_linear_solver=[]
class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res
   
def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon
     
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0

def analyze(nn, LB_N0, UB_N0, label, layer_pattern):    
    LinearSolver = layer_pattern #True# editable
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    # define interval array from perturbed img
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ##******* ELINA construct input abstraction from interval array *******
    # input zonotope (box), no int variable, all real variable
    element = elina_abstract0_of_box(man, 0, num_pixels, itv) #abstraction, create a hypercude defiend by the ElinaIntervalArray, 

    # free interval array
    elina_interval_array_free(itv,num_pixels)
    
    # go through each layer
    for layerno in range(numlayer):
        # for each layer
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           # get current dimention of element
           dims = elina_abstract0_dimension(man,element) #element: abstraction of box
           num_in_pixels = dims.intdim + dims.realdim
           # get target dimension
           num_out_pixels = len(weights)
           
           # construct dimension to be added
           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)

           # weights to contiguousarray
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels

           #******* Linear solver model for affine *******
           m = Model("Relu")
           in_bounds = elina_abstract0_to_box(man,element)
           
           # Add input variable to linear solver as x1-xn
           in_var_list=[]
           for i in range(num_in_pixels):
               in_lb=in_bounds[i].contents.inf.contents.val.dbl
               in_ub=in_bounds[i].contents.sup.contents.val.dbl
               var_str="x{}".format(i)
               in_var_list.append( m.addVar(lb=in_lb,ub=in_ub,name=var_str) )
           m.update() #update model so that linear expression can access var name
           elina_interval_array_free(in_bounds,var)
           #******* ELINA For affine *******
           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)

           #******* If ReLU *******
           
           LB_lin=[]
           UB_lin=[]
           if(nn.layertypes[layerno]=='ReLU'): 
               ## ELINA
               if (not LinearSolver[layerno]):
                   element = relu_box_layerwise(man,True,element,0, num_out_pixels) # (man,desctructive,elem,start_offset,num_dimension)
               else:
                   layers_with_linear_solver.append(layerno)
               ## Linear Solver
                   activat0_bounds = elina_abstract0_to_box(man,element)
                   itv_lin = elina_interval_array_alloc(num_out_pixels)
                   for i in range(num_out_pixels):
                       a_lb =activat0_bounds[i].contents.inf.contents.val.dbl
                       b_ub =activat0_bounds[i].contents.sup.contents.val.dbl
                       # expression for the activation of an neuron
                       activat0_linexpr0=LinExpr(weights[i].tolist(),in_var_list)
                       print(biases[i])
                       activat0_linexpr0.addConstant(biases[i])
                       print(activat0_linexpr0)
                       # define relu output as y1-yn and set constraint
                       var_str="y{}".format(i)
                       y=[]
                       if(a_lb>=0):
                           y.append(m.addVar(lb=a_lb,ub=b_ub,name=var_str))
                       elif(b_ub<=0):
                           y.append(m.addVar(lb=0.0,ub=0.0,name=var_str))
                       else:
                           grad_lin = b_ub/(b_ub-a_lb)
                           bias_lin = -b_ub*a_lb/(b_ub-a_lb)
                           #print(grad_lin,bias_lin)
                           y.append(m.addVar(name=var_str))
                           m.addConstr(y[0]>= 0)
                           #print(activat0_linexpr0)
                           m.addConstr(y[0]>= activat0_linexpr0)
                           m.addConstr((y[0]-bias_lin)/grad_lin<= activat0_linexpr0)
                       m.update()
                       # minimise for lower bound
                       m.setObjective(y[0],GRB.MINIMIZE)
                       m.optimize()
                       y_lb=y[0].x
                       # maximise for upper bound
                       m.setObjective(y[0],GRB.MAXIMIZE)
                       m.optimize()
                       y_ub=y[0].x
                       print('layer{},neuron{},lb:{},ub:{}'.format(layerno,i,y_lb,y_ub))
                       # define interval array from perturbed img
                       elina_interval_set_double(itv_lin[i],y_lb,y_ub)
                   
                       ##******* ELINA construct input abstraction from interval array *******
                   element = elina_abstract0_of_box(man, 0, num_out_pixels, itv_lin) 
                   elina_interval_array_free(itv_lin,num_out_pixels)
                   elina_interval_array_free(activat0_bounds,num_out_pixels)
           nn.ffn_counter+=1 

        else:
           print(' net type not supported')
   
    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)

           
    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]): # LB=UB => epsilon =0 
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break    
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    # other label has higher chance than true label
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, verified_flag

#for the switch to have certain pattern for corresponding network
#currently just a sample 
def switch(netname):
    return {
        '3_10':[True, True, True],
        '3_20':[True, True, True],
    }[netname]
#*******************************************

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)#check nn.numlayer
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    
#####****************setup layer pattern
    #all_linear_pattern = []
    #for i in range(nn.numlayer):
        #all_linear_pattern.append
    sample_linear_pattern = [True for _ in range(nn.numlayer)]#setup linear solver pattern
    
######******************88888

    label, _ = analyze(nn,LB_N0,UB_N0,0,linear_pattern)
    start = time.time()
    is_valid = False
    is_verified =False
    if(label==int(x0_low[0])):
        is_valid = True
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label,sample_linear_pattern)# add layer specification
        is_verified=verified_flag
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        is_valid = False
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
     
    ##*********OUTPUT data**********##
    csv_address='/home/riai2018/RIAI_pjct/output/log.csv'
    exists = os.path.isfile(csv_address)
    mode = 'a' if (exists) else 'w'
    fields=['network','image','epsilon','is_valid','is_verified','layers_with_linear_solver','time']
    with open('/home/riai2018/RIAI_pjct/output/log.csv',mode) as f:
        w =csv.writer(f)
        if (not exists):
            w.writerow(fields)
        w.writerow([re.search(r"\d+_\d+",netname).group(),re.search(r"img\d+",specname).group(),epsilon,is_valid,is_verified,linear_pattern,end-start])



