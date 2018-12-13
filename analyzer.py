##########################################################
# Code Strcture
#1. For each network, a pattern list specifies whether linear solver is used in each layer
#   eg.[True,True,False,False]
#2. Elina is used in all layers for rough lower and upper bound
#
#3. 1) At the beginning of each layer, Elina handles affine transformation
#   2) If it is a ReLu layer, either Elina or Gorubi handles ReLu depending on the pattern specified
#   3) If Gorubi was used, a new Elina abstraction is constructed after obtaining the bounds. So we always end up with an Elina box
#
#
#
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

def toElina(lb,ub,man,num_pixels):
    ##******* ELINA construct input abstraction from interval array *******
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],lb[i],ub[i])
    # input zonotope (box), no int variable, all real variable
    element = elina_abstract0_of_box(man, 0, num_pixels, itv) #abstraction, create a hypercude defiend by the ElinaIntervalArray, 
    elina_interval_array_free(itv,num_pixels)
    return element

def getVarListfromElina(model,element,man,num_pixels,layerno):
    var_list=[]
    lb=[]
    ub=[]
    bounds = elina_abstract0_to_box(man,element)
    for i in range(num_pixels):
        lb.append(bounds[i].contents.inf.contents.val.dbl)
        ub.append(bounds[i].contents.sup.contents.val.dbl)
        var_str="y{}_{}".format(layerno,i)
        var_list.append( model.addVar(lb=lb[i],ub=ub[i],name=var_str) )
    model.update() #update model so that linear expression can access var name
    elina_interval_array_free(bounds,num_pixels)
    return var_list

def getBoundsFromElina(element,man,num_pixels):
    lb=[]
    ub=[]
    bounds = elina_abstract0_to_box(man,element)
    for i in range(num_pixels):
        lb.append(bounds[i].contents.inf.contents.val.dbl)
        ub.append(bounds[i].contents.sup.contents.val.dbl)
    elina_interval_array_free(bounds,num_pixels)
    return lb,ub

def analyze(nn, LB_N0, UB_N0, label, layer_pattern):    
    LinearSolver = layer_pattern #True# editable
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer
    ## initialise Elina ##
    man = elina_box_manager_alloc()
    element = toElina(LB_N0,UB_N0,man,num_pixels)
    ## initiaise Linear solver ##
    m = Model("Gorubi")
    m.Params.OutputFlag=0
    # dictionary to hold all gurobi linear expressions,{layerno:[expr_list]} ##
    expr0_dict={}
    # dictionay to hold all gurobi variables, {layerno:[input_var_list]}. Note img is the input to layer0 
    img_var_list= getVarListfromElina(m,element,man,num_pixels,0)
    var_dict={0:img_var_list}

    # go through each layer
    for layerno in range(numlayer):
        # for each layer
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            #******* ELINA For affine *******
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
            # ****** Linear solver affine *******
            if ( LinearSolver[layerno] and (LB_N0[0]!=UB_N0[0]) ):
                expr0_list=[]
                for i in range(num_out_pixels):
                    activat0_linexpr0=LinExpr(weights[i].tolist(),var_dict[layerno])
                    activat0_linexpr0.addConstant(biases[i])
                    expr0_list.append(activat0_linexpr0)  
                expr0_dict.update({layerno:expr0_list})

            #******* If ReLU *******
            if(nn.layertypes[layerno]=='ReLU'): 
                ## ELINA
                if ((not LinearSolver[layerno]) or (LB_N0[0]==UB_N0[0])):
                     element = relu_box_layerwise(man,True,element,0, num_out_pixels) # (man,desctructive,elem,start_offset,num_dimension)
                     # Get list of variable for next layer (may or may not be used)
                     if (not LB_N0[0]==UB_N0[0]):
                       var_list= getVarListfromElina(m,element,man,num_out_pixels,layerno)
                       var_dict.update({layerno+1:var_list})
                ## Linear Solver
                else:
                     layers_with_linear_solver.append(layerno)
                     a_lb_array,b_ub_array = getBoundsFromElina(element,man,num_out_pixels)
                     # List of output variable of the layer
                     y=[]
                     # For each output neuron, perform ReLu approximation
                     for i in range(num_out_pixels):
                          a_lb=a_lb_array[i]
                          b_ub=b_ub_array[i]
                          var_str="y{}_{}".format(layerno,i)
                          if(a_lb>=0):
                              y.append(m.addVar(name=var_str))
                              m.addConstr(y[i]>= a_lb) 
                              m.addConstr(y[i]<= b_ub) 
                              # Relu(h) = h
                              m.addConstr(y[i]== expr0_dict[layerno][i] )
                          elif(b_ub<=0):
                              y.append(m.addVar(name=var_str))
                              # Relu(h) = 0
                              m.addConstr(y[i]== 0) 
                          else:
                              grad_lin = b_ub/(b_ub-a_lb)
                              bias_lin = -b_ub*a_lb/(b_ub-a_lb)
                              y.append(m.addVar(name=var_str))
                              # Relu(h) = grad*h+bias
                              m.addConstr(y[i]>= 0) 
                              m.addConstr(y[i]>= expr0_dict[layerno][i] )
                              m.addConstr(y[i]<= grad_lin*expr0_dict[layerno][i]+bias_lin) 
                     # Gather var list y as input for the next layer (may or may not be used)
                     var_dict.update({layerno+1:y})
                     m.update()
                     # Empty as place holder for final bounds of the layer
                     y_lb=[]
                     y_ub=[]
                     for i in range(num_out_pixels):
                         # minimise for lower bound
                         m.setObjective(y[i],GRB.MINIMIZE)
                         m.update()
                         m.optimize()
                         y_lb.append(y[i].x)
                         # maximise for upper bound
                         m.setObjective(y[i],GRB.MAXIMIZE)
                         m.update()
                         m.optimize()
                         y_ub.append(y[i].x)
                     # Construct Elina Box    
                     element = toElina(y_lb,y_ub,man,num_out_pixels) 
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
        #'3_10':[False,False,False],
        '3_10':[True, True, True],
        '3_20':[True, True, False],#3*true:90 tft:33 ftt:71 ttf:81
        '3_50':[True, True, True],
        '4_1024':[True, True, True, True],
        '6_20':[True, True, True, True, True, True],
        '6_50':[True, True, True, True, True, True],
        '6_100':[True, True, True, False, False, False],
        #'6_100':[False,False,False,False,False,False,],
        '6_200':[True, True, True, True, True, True],
        '9_100':[True, True, True, True, True, True, True, True, True],
        '9_200':[True, True, True, True, True, True, True, True, True],
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
    ##possible usage of switch to choose suitable pattern
    linear_pattern = switch(re.search(r"\d+_\d+",netname).group())

    #sample_linear_pattern = [True for _ in range(nn.numlayer)]#setup linear solver pattern
    
######******************88888

    label, _ = analyze(nn,LB_N0,UB_N0,0,linear_pattern)
    start = time.time()
    is_valid = False
    is_verified =False
    if(label==int(x0_low[0])):
        is_valid = True
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label,linear_pattern)# add layer specification
        is_verified=verified_flag
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
        print("True Label {}".format(x0_low[0]))  
    else:
        is_valid = False
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
     
    ##*********OUTPUT data**********##
    #csv_address='/home/riai2018/RIAI_pjct/output/log.csv'
    csv_address='log.csv'
    exists = os.path.isfile(csv_address)
    mode = 'a' if (exists) else 'w'
    fields=['network','image','epsilon','is_valid','is_verified','layers_with_linear_solver','time']
    with open(csv_address,mode) as f:
        w =csv.writer(f)
        if (not exists):
            w.writerow(fields)
        w.writerow([re.search(r"\d+_\d+",netname).group(),re.search(r"img\d+",specname).group(),epsilon,is_valid,is_verified,linear_pattern,end-start])



