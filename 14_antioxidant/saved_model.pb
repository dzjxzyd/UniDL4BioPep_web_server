Ç
Î
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Öù

SGD/fc2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameSGD/fc2/bias/momentum
{
)SGD/fc2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/fc2/bias/momentum*
_output_shapes
:*
dtype0

SGD/fc2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameSGD/fc2/kernel/momentum

+SGD/fc2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc2/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/fc1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameSGD/fc1/bias/momentum
{
)SGD/fc1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/fc1/bias/momentum*
_output_shapes
:@*
dtype0

SGD/fc1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(@*(
shared_nameSGD/fc1/kernel/momentum

+SGD/fc1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc1/kernel/momentum*
_output_shapes
:	(@*
dtype0
¨
(SGD/batch_normalization_47/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_47/beta/momentum
¡
<SGD/batch_normalization_47/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_47/beta/momentum*
_output_shapes
: *
dtype0
ª
)SGD/batch_normalization_47/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_47/gamma/momentum
£
=SGD/batch_normalization_47/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_47/gamma/momentum*
_output_shapes
: *
dtype0

SGD/layer_conv2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/layer_conv2/bias/momentum

1SGD/layer_conv2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer_conv2/bias/momentum*
_output_shapes
: *
dtype0

SGD/layer_conv2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!SGD/layer_conv2/kernel/momentum

3SGD/layer_conv2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer_conv2/kernel/momentum*"
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
h
fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc2/bias
a
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
_output_shapes
:*
dtype0
p

fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
fc2/kernel
i
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel*
_output_shapes

:@*
dtype0
h
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
fc1/bias
a
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes
:@*
dtype0
q

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(@*
shared_name
fc1/kernel
j
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
_output_shapes
:	(@*
dtype0
¤
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_47/moving_variance

:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_47/moving_mean

6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_47/beta

/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes
: *
dtype0

batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_47/gamma

0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes
: *
dtype0
x
layer_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namelayer_conv2/bias
q
$layer_conv2/bias/Read/ReadVariableOpReadVariableOplayer_conv2/bias*
_output_shapes
: *
dtype0

layer_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namelayer_conv2/kernel
}
&layer_conv2/kernel/Read/ReadVariableOpReadVariableOplayer_conv2/kernel*"
_output_shapes
: *
dtype0

serving_default_input_24Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿÀ
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_24layer_conv2/kernellayer_conv2/bias&batch_normalization_47/moving_variancebatch_normalization_47/gamma"batch_normalization_47/moving_meanbatch_normalization_47/beta
fc1/kernelfc1/bias
fc2/kernelfc2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_617717

NoOpNoOp
óJ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*®J
value¤JB¡J BJ
¶
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
È
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
Õ
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#axis
	$gamma
%beta
&moving_mean
'moving_variance*

(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 

.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
¥
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_random_generator* 

;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
¦
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias*
¥
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator* 
¦
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
J
0
1
$2
%3
&4
'5
G6
H7
V8
W9*
<
0
1
$2
%3
G4
H5
V6
W7*
* 
°
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
]trace_0
^trace_1
_trace_2
`trace_3* 
6
atrace_0
btrace_1
ctrace_2
dtrace_3* 
* 
Â
	edecay
flearning_rate
gmomentum
hitermomentumºmomentum»$momentum¼%momentum½Gmomentum¾Hmomentum¿VmomentumÀWmomentumÁ*

iserving_default* 

0
1*

0
1*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
b\
VARIABLE_VALUElayer_conv2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUElayer_conv2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
$0
%1
&2
'3*

$0
%1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_47/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_47/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_47/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_47/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

G0
H1*

G0
H1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ZT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

¤trace_0
¥trace_1* 

¦trace_0
§trace_1* 
* 

V0
W1*

V0
W1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

­trace_0* 

®trace_0* 
ZT
VARIABLE_VALUE
fc2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*
J
0
1
2
3
4
5
6
7
	8

9*

¯0
°1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
±	variables
²	keras_api

³total

´count*
M
µ	variables
¶	keras_api

·total

¸count
¹
_fn_kwargs*

³0
´1*

±	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

·0
¸1*

µ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUESGD/layer_conv2/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/layer_conv2/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)SGD/batch_normalization_47/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(SGD/batch_normalization_47/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/fc1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/fc1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/fc2/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/fc2/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Õ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&layer_conv2/kernel/Read/ReadVariableOp$layer_conv2/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3SGD/layer_conv2/kernel/momentum/Read/ReadVariableOp1SGD/layer_conv2/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_47/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_47/beta/momentum/Read/ReadVariableOp+SGD/fc1/kernel/momentum/Read/ReadVariableOp)SGD/fc1/bias/momentum/Read/ReadVariableOp+SGD/fc2/kernel/momentum/Read/ReadVariableOp)SGD/fc2/bias/momentum/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_618236
È
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_conv2/kernellayer_conv2/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/biasdecaylearning_ratemomentumSGD/itertotal_1count_1totalcountSGD/layer_conv2/kernel/momentumSGD/layer_conv2/bias/momentum)SGD/batch_normalization_47/gamma/momentum(SGD/batch_normalization_47/beta/momentumSGD/fc1/kernel/momentumSGD/fc1/bias/momentumSGD/fc2/kernel/momentumSGD/fc2/bias/momentum*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_618324ëä


ñ
?__inference_fc1_layer_call_and_return_conditional_losses_617370

inputs1
matmul_readvariableop_resource:	(@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs


ì
(__inference_Predict_layer_call_fn_617620
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24
Ù
d
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í
d
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


ñ
?__inference_fc1_layer_call_and_return_conditional_losses_618088

inputs1
matmul_readvariableop_resource:	(@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
÷	
è
$__inference_signature_wrapper_617717
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_617202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24
í
d
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ü
Ò
7__inference_batch_normalization_47_layer_call_fn_617953

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ð
?__inference_fc2_layer_call_and_return_conditional_losses_617394

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ù

G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
­
G
+__inference_flatten_23_layer_call_fn_618062

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ó
d
+__inference_dropout_71_layer_call_fn_618098

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¡
G
+__inference_dropout_71_layer_call_fn_618093

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¡C
ÿ
C__inference_Predict_layer_call_and_return_conditional_losses_617821

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: F
8batch_normalization_47_batchnorm_readvariableop_resource: J
<batch_normalization_47_batchnorm_mul_readvariableop_resource: H
:batch_normalization_47_batchnorm_readvariableop_1_resource: H
:batch_normalization_47_batchnorm_readvariableop_2_resource: 5
"fc1_matmul_readvariableop_resource:	(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identity¢/batch_normalization_47/batchnorm/ReadVariableOp¢1batch_normalization_47/batchnorm/ReadVariableOp_1¢1batch_normalization_47/batchnorm/ReadVariableOp_2¢3batch_normalization_47/batchnorm/mul/ReadVariableOp¢fc1/BiasAdd/ReadVariableOp¢fc1/MatMul/ReadVariableOp¢fc2/BiasAdd/ReadVariableOp¢fc2/MatMul/ReadVariableOp¢"layer_conv2/BiasAdd/ReadVariableOp¢.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpl
!layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
layer_conv2/Conv1D/ExpandDims
ExpandDimsinputs*layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀª
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0e
#layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ä
layer_conv2/Conv1D/ExpandDims_1
ExpandDims6layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ñ
layer_conv2/Conv1DConv2D&layer_conv2/Conv1D/ExpandDims:output:0(layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
paddingSAME*
strides

layer_conv2/Conv1D/SqueezeSqueezelayer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¦
layer_conv2/BiasAddBiasAdd#layer_conv2/Conv1D/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¤
/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_47/batchnorm/addAddV27batch_normalization_47/batchnorm/ReadVariableOp:value:0/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_47/batchnorm/RsqrtRsqrt(batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: ¬
3batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0¹
$batch_normalization_47/batchnorm/mulMul*batch_normalization_47/batchnorm/Rsqrt:y:0;batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: ¬
&batch_normalization_47/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¨
1batch_normalization_47/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_47_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0·
&batch_normalization_47/batchnorm/mul_2Mul9batch_normalization_47/batchnorm/ReadVariableOp_1:value:0(batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: ¨
1batch_normalization_47/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_47_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0·
$batch_normalization_47/batchnorm/subSub9batch_normalization_47/batchnorm/ReadVariableOp_2:value:0*batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¼
&batch_normalization_47/batchnorm/add_1AddV2*batch_normalization_47/batchnorm/mul_1:z:0(batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ }
activation_47/ReluRelu*batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ Y
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
MaxPool2/ExpandDims
ExpandDims activation_47/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¦
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingSAME*
strides

MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
q
dropout_70/IdentityIdentityMaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  a
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_23/ReshapeReshapedropout_70/Identity:output:0flatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype0

fc1/MatMulMatMulflatten_23/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout_71/IdentityIdentityfc1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0

fc2/MatMulMatMuldropout_71/Identity:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp0^batch_normalization_47/batchnorm/ReadVariableOp2^batch_normalization_47/batchnorm/ReadVariableOp_12^batch_normalization_47/batchnorm/ReadVariableOp_24^batch_normalization_47/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2b
/batch_normalization_47/batchnorm/ReadVariableOp/batch_normalization_47/batchnorm/ReadVariableOp2f
1batch_normalization_47/batchnorm/ReadVariableOp_11batch_normalization_47/batchnorm/ReadVariableOp_12f
1batch_normalization_47/batchnorm/ReadVariableOp_21batch_normalization_47/batchnorm/ReadVariableOp_22j
3batch_normalization_47/batchnorm/mul/ReadVariableOp3batch_normalization_47/batchnorm/mul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2H
"layer_conv2/BiasAdd/ReadVariableOp"layer_conv2/BiasAdd/ReadVariableOp2`
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
»

$__inference_fc1_layer_call_fn_618077

inputs
unknown:	(@
	unknown_0:@
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
á
e
I__inference_activation_47_layer_call_and_return_conditional_losses_618017

inputs
identityK
ReluReluinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ _
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
 
_user_specified_nameinputs
È
`
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
b
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


ì
(__inference_Predict_layer_call_fn_617424
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24
ôK
ÿ	
!__inference__wrapped_model_617202
input_24U
?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource: A
3predict_layer_conv2_biasadd_readvariableop_resource: N
@predict_batch_normalization_47_batchnorm_readvariableop_resource: R
Dpredict_batch_normalization_47_batchnorm_mul_readvariableop_resource: P
Bpredict_batch_normalization_47_batchnorm_readvariableop_1_resource: P
Bpredict_batch_normalization_47_batchnorm_readvariableop_2_resource: =
*predict_fc1_matmul_readvariableop_resource:	(@9
+predict_fc1_biasadd_readvariableop_resource:@<
*predict_fc2_matmul_readvariableop_resource:@9
+predict_fc2_biasadd_readvariableop_resource:
identity¢7Predict/batch_normalization_47/batchnorm/ReadVariableOp¢9Predict/batch_normalization_47/batchnorm/ReadVariableOp_1¢9Predict/batch_normalization_47/batchnorm/ReadVariableOp_2¢;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOp¢"Predict/fc1/BiasAdd/ReadVariableOp¢!Predict/fc1/MatMul/ReadVariableOp¢"Predict/fc2/BiasAdd/ReadVariableOp¢!Predict/fc2/MatMul/ReadVariableOp¢*Predict/layer_conv2/BiasAdd/ReadVariableOp¢6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpt
)Predict/layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¬
%Predict/layer_conv2/Conv1D/ExpandDims
ExpandDimsinput_242Predict/layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀº
6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0m
+Predict/layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ü
'Predict/layer_conv2/Conv1D/ExpandDims_1
ExpandDims>Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:04Predict/layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: é
Predict/layer_conv2/Conv1DConv2D.Predict/layer_conv2/Conv1D/ExpandDims:output:00Predict/layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
paddingSAME*
strides
©
"Predict/layer_conv2/Conv1D/SqueezeSqueeze#Predict/layer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
*Predict/layer_conv2/BiasAdd/ReadVariableOpReadVariableOp3predict_layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¾
Predict/layer_conv2/BiasAddBiasAdd+Predict/layer_conv2/Conv1D/Squeeze:output:02Predict/layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ´
7Predict/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp@predict_batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.Predict/batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ô
,Predict/batch_normalization_47/batchnorm/addAddV2?Predict/batch_normalization_47/batchnorm/ReadVariableOp:value:07Predict/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: 
.Predict/batch_normalization_47/batchnorm/RsqrtRsqrt0Predict/batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: ¼
;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOpDpredict_batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Ñ
,Predict/batch_normalization_47/batchnorm/mulMul2Predict/batch_normalization_47/batchnorm/Rsqrt:y:0CPredict/batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: Ä
.Predict/batch_normalization_47/batchnorm/mul_1Mul$Predict/layer_conv2/BiasAdd:output:00Predict/batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¸
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_1ReadVariableOpBpredict_batch_normalization_47_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0Ï
.Predict/batch_normalization_47/batchnorm/mul_2MulAPredict/batch_normalization_47/batchnorm/ReadVariableOp_1:value:00Predict/batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: ¸
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_2ReadVariableOpBpredict_batch_normalization_47_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0Ï
,Predict/batch_normalization_47/batchnorm/subSubAPredict/batch_normalization_47/batchnorm/ReadVariableOp_2:value:02Predict/batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: Ô
.Predict/batch_normalization_47/batchnorm/add_1AddV22Predict/batch_normalization_47/batchnorm/mul_1:z:00Predict/batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
Predict/activation_47/ReluRelu2Predict/batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ a
Predict/MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¸
Predict/MaxPool2/ExpandDims
ExpandDims(Predict/activation_47/Relu:activations:0(Predict/MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¶
Predict/MaxPool2/MaxPoolMaxPool$Predict/MaxPool2/ExpandDims:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingSAME*
strides

Predict/MaxPool2/SqueezeSqueeze!Predict/MaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

Predict/dropout_70/IdentityIdentity!Predict/MaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  i
Predict/flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¡
Predict/flatten_23/ReshapeReshape$Predict/dropout_70/Identity:output:0!Predict/flatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!Predict/fc1/MatMul/ReadVariableOpReadVariableOp*predict_fc1_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype0
Predict/fc1/MatMulMatMul#Predict/flatten_23/Reshape:output:0)Predict/fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"Predict/fc1/BiasAdd/ReadVariableOpReadVariableOp+predict_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Predict/fc1/BiasAddBiasAddPredict/fc1/MatMul:product:0*Predict/fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
Predict/fc1/ReluReluPredict/fc1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
Predict/dropout_71/IdentityIdentityPredict/fc1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!Predict/fc2/MatMul/ReadVariableOpReadVariableOp*predict_fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
Predict/fc2/MatMulMatMul$Predict/dropout_71/Identity:output:0)Predict/fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Predict/fc2/BiasAdd/ReadVariableOpReadVariableOp+predict_fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Predict/fc2/BiasAddBiasAddPredict/fc2/MatMul:product:0*Predict/fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Predict/fc2/SoftmaxSoftmaxPredict/fc2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentityPredict/fc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp8^Predict/batch_normalization_47/batchnorm/ReadVariableOp:^Predict/batch_normalization_47/batchnorm/ReadVariableOp_1:^Predict/batch_normalization_47/batchnorm/ReadVariableOp_2<^Predict/batch_normalization_47/batchnorm/mul/ReadVariableOp#^Predict/fc1/BiasAdd/ReadVariableOp"^Predict/fc1/MatMul/ReadVariableOp#^Predict/fc2/BiasAdd/ReadVariableOp"^Predict/fc2/MatMul/ReadVariableOp+^Predict/layer_conv2/BiasAdd/ReadVariableOp7^Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2r
7Predict/batch_normalization_47/batchnorm/ReadVariableOp7Predict/batch_normalization_47/batchnorm/ReadVariableOp2v
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_19Predict/batch_normalization_47/batchnorm/ReadVariableOp_12v
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_29Predict/batch_normalization_47/batchnorm/ReadVariableOp_22z
;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOp;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOp2H
"Predict/fc1/BiasAdd/ReadVariableOp"Predict/fc1/BiasAdd/ReadVariableOp2F
!Predict/fc1/MatMul/ReadVariableOp!Predict/fc1/MatMul/ReadVariableOp2H
"Predict/fc2/BiasAdd/ReadVariableOp"Predict/fc2/BiasAdd/ReadVariableOp2F
!Predict/fc2/MatMul/ReadVariableOp!Predict/fc2/MatMul/ReadVariableOp2X
*Predict/layer_conv2/BiasAdd/ReadVariableOp*Predict/layer_conv2/BiasAdd/ReadVariableOp2p
6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24


e
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á
e
I__inference_activation_47_layer_call_and_return_conditional_losses_617341

inputs
identityK
ReluReluinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ _
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
 
_user_specified_nameinputs
Ã)
°
C__inference_Predict_layer_call_and_return_conditional_losses_617572

inputs(
layer_conv2_617542:  
layer_conv2_617544: +
batch_normalization_47_617547: +
batch_normalization_47_617549: +
batch_normalization_47_617551: +
batch_normalization_47_617553: 

fc1_617560:	(@

fc1_617562:@

fc2_617566:@

fc2_617568:
identity¢.batch_normalization_47/StatefulPartitionedCall¢"dropout_70/StatefulPartitionedCall¢"dropout_71/StatefulPartitionedCall¢fc1/StatefulPartitionedCall¢fc2/StatefulPartitionedCall¢#layer_conv2/StatefulPartitionedCall
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_617542layer_conv2_617544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617547batch_normalization_47_617549batch_normalization_47_617551batch_normalization_47_617553*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273ø
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341Ý
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296ì
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493â
flatten_23/PartitionedCallPartitionedCall+dropout_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357ù
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0
fc1_617560
fc1_617562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370
"dropout_71/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_70/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_71/StatefulPartitionedCall:output:0
fc2_617566
fc2_617568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall#^dropout_71/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2H
"dropout_71/StatefulPartitionedCall"dropout_71/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
È
`
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þ%
ë
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
m
ç	
C__inference_Predict_layer_call_and_return_conditional_losses_617903

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: L
>batch_normalization_47_assignmovingavg_readvariableop_resource: N
@batch_normalization_47_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_47_batchnorm_mul_readvariableop_resource: F
8batch_normalization_47_batchnorm_readvariableop_resource: 5
"fc1_matmul_readvariableop_resource:	(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identity¢&batch_normalization_47/AssignMovingAvg¢5batch_normalization_47/AssignMovingAvg/ReadVariableOp¢(batch_normalization_47/AssignMovingAvg_1¢7batch_normalization_47/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_47/batchnorm/ReadVariableOp¢3batch_normalization_47/batchnorm/mul/ReadVariableOp¢fc1/BiasAdd/ReadVariableOp¢fc1/MatMul/ReadVariableOp¢fc2/BiasAdd/ReadVariableOp¢fc2/MatMul/ReadVariableOp¢"layer_conv2/BiasAdd/ReadVariableOp¢.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpl
!layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
layer_conv2/Conv1D/ExpandDims
ExpandDimsinputs*layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀª
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0e
#layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ä
layer_conv2/Conv1D/ExpandDims_1
ExpandDims6layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ñ
layer_conv2/Conv1DConv2D&layer_conv2/Conv1D/ExpandDims:output:0(layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
paddingSAME*
strides

layer_conv2/Conv1D/SqueezeSqueezelayer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¦
layer_conv2/BiasAddBiasAdd#layer_conv2/Conv1D/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
5batch_normalization_47/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ç
#batch_normalization_47/moments/meanMeanlayer_conv2/BiasAdd:output:0>batch_normalization_47/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
+batch_normalization_47/moments/StopGradientStopGradient,batch_normalization_47/moments/mean:output:0*
T0*"
_output_shapes
: Ð
0batch_normalization_47/moments/SquaredDifferenceSquaredDifferencelayer_conv2/BiasAdd:output:04batch_normalization_47/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
9batch_normalization_47/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ç
'batch_normalization_47/moments/varianceMean4batch_normalization_47/moments/SquaredDifference:z:0Bbatch_normalization_47/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
&batch_normalization_47/moments/SqueezeSqueeze,batch_normalization_47/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 ¢
(batch_normalization_47/moments/Squeeze_1Squeeze0batch_normalization_47/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 q
,batch_normalization_47/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_47/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_47_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Æ
*batch_normalization_47/AssignMovingAvg/subSub=batch_normalization_47/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_47/moments/Squeeze:output:0*
T0*
_output_shapes
: ½
*batch_normalization_47/AssignMovingAvg/mulMul.batch_normalization_47/AssignMovingAvg/sub:z:05batch_normalization_47/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 
&batch_normalization_47/AssignMovingAvgAssignSubVariableOp>batch_normalization_47_assignmovingavg_readvariableop_resource.batch_normalization_47/AssignMovingAvg/mul:z:06^batch_normalization_47/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_47/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_47/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_47_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Ì
,batch_normalization_47/AssignMovingAvg_1/subSub?batch_normalization_47/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_47/moments/Squeeze_1:output:0*
T0*
_output_shapes
: Ã
,batch_normalization_47/AssignMovingAvg_1/mulMul0batch_normalization_47/AssignMovingAvg_1/sub:z:07batch_normalization_47/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 
(batch_normalization_47/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_47_assignmovingavg_1_readvariableop_resource0batch_normalization_47/AssignMovingAvg_1/mul:z:08^batch_normalization_47/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_47/batchnorm/addAddV21batch_normalization_47/moments/Squeeze_1:output:0/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_47/batchnorm/RsqrtRsqrt(batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: ¬
3batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0¹
$batch_normalization_47/batchnorm/mulMul*batch_normalization_47/batchnorm/Rsqrt:y:0;batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: ¬
&batch_normalization_47/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ­
&batch_normalization_47/batchnorm/mul_2Mul/batch_normalization_47/moments/Squeeze:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: ¤
/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0µ
$batch_normalization_47/batchnorm/subSub7batch_normalization_47/batchnorm/ReadVariableOp:value:0*batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¼
&batch_normalization_47/batchnorm/add_1AddV2*batch_normalization_47/batchnorm/mul_1:z:0(batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ }
activation_47/ReluRelu*batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ Y
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
MaxPool2/ExpandDims
ExpandDims activation_47/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ ¦
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingSAME*
strides

MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims
]
dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_70/dropout/MulMulMaxPool2/Squeeze:output:0!dropout_70/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  a
dropout_70/dropout/ShapeShapeMaxPool2/Squeeze:output:0*
T0*
_output_shapes
:§
/dropout_70/dropout/random_uniform/RandomUniformRandomUniform!dropout_70/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0f
!dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ì
dropout_70/dropout/GreaterEqualGreaterEqual8dropout_70/dropout/random_uniform/RandomUniform:output:0*dropout_70/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  _
dropout_70/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
dropout_70/dropout/SelectV2SelectV2#dropout_70/dropout/GreaterEqual:z:0dropout_70/dropout/Mul:z:0#dropout_70/dropout/Const_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  a
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_23/ReshapeReshape$dropout_70/dropout/SelectV2:output:0flatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	(@*
dtype0

fc1/MatMulMatMulflatten_23/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dropout_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
dropout_71/dropout/MulMulfc1/Relu:activations:0!dropout_71/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
dropout_71/dropout/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:¢
/dropout_71/dropout/random_uniform/RandomUniformRandomUniform!dropout_71/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0f
!dropout_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
dropout_71/dropout/GreaterEqualGreaterEqual8dropout_71/dropout/random_uniform/RandomUniform:output:0*dropout_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dropout_71/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ¿
dropout_71/dropout/SelectV2SelectV2#dropout_71/dropout/GreaterEqual:z:0dropout_71/dropout/Mul:z:0#dropout_71/dropout/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0

fc2/MatMulMatMul$dropout_71/dropout/SelectV2:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp'^batch_normalization_47/AssignMovingAvg6^batch_normalization_47/AssignMovingAvg/ReadVariableOp)^batch_normalization_47/AssignMovingAvg_18^batch_normalization_47/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_47/batchnorm/ReadVariableOp4^batch_normalization_47/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2P
&batch_normalization_47/AssignMovingAvg&batch_normalization_47/AssignMovingAvg2n
5batch_normalization_47/AssignMovingAvg/ReadVariableOp5batch_normalization_47/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_47/AssignMovingAvg_1(batch_normalization_47/AssignMovingAvg_12r
7batch_normalization_47/AssignMovingAvg_1/ReadVariableOp7batch_normalization_47/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_47/batchnorm/ReadVariableOp/batch_normalization_47/batchnorm/ReadVariableOp2j
3batch_normalization_47/batchnorm/mul/ReadVariableOp3batch_normalization_47/batchnorm/mul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2H
"layer_conv2/BiasAdd/ReadVariableOp"layer_conv2/BiasAdd/ReadVariableOp2`
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs


ð
?__inference_fc2_layer_call_and_return_conditional_losses_618135

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò;
ª
__inference__traced_save_618236
file_prefix1
-savev2_layer_conv2_kernel_read_readvariableop/
+savev2_layer_conv2_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop)
%savev2_fc2_kernel_read_readvariableop'
#savev2_fc2_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_sgd_layer_conv2_kernel_momentum_read_readvariableop<
8savev2_sgd_layer_conv2_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_47_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_47_beta_momentum_read_readvariableop6
2savev2_sgd_fc1_kernel_momentum_read_readvariableop4
0savev2_sgd_fc1_bias_momentum_read_readvariableop6
2savev2_sgd_fc2_kernel_momentum_read_readvariableop4
0savev2_sgd_fc2_bias_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
:  
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*É
value¿B¼B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH£
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B È
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_layer_conv2_kernel_read_readvariableop+savev2_layer_conv2_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_sgd_layer_conv2_kernel_momentum_read_readvariableop8savev2_sgd_layer_conv2_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_47_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_47_beta_momentum_read_readvariableop2savev2_sgd_fc1_kernel_momentum_read_readvariableop0savev2_sgd_fc1_bias_momentum_read_readvariableop2savev2_sgd_fc2_kernel_momentum_read_readvariableop0savev2_sgd_fc2_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: : : : : : : :	(@:@:@:: : : : : : : : : : : : :	(@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	(@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	(@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
É)
²
C__inference_Predict_layer_call_and_return_conditional_losses_617686
input_24(
layer_conv2_617656:  
layer_conv2_617658: +
batch_normalization_47_617661: +
batch_normalization_47_617663: +
batch_normalization_47_617665: +
batch_normalization_47_617667: 

fc1_617674:	(@

fc1_617676:@

fc2_617680:@

fc2_617682:
identity¢.batch_normalization_47/StatefulPartitionedCall¢"dropout_70/StatefulPartitionedCall¢"dropout_71/StatefulPartitionedCall¢fc1/StatefulPartitionedCall¢fc2/StatefulPartitionedCall¢#layer_conv2/StatefulPartitionedCall
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_24layer_conv2_617656layer_conv2_617658*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617661batch_normalization_47_617663batch_normalization_47_617665batch_normalization_47_617667*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273ø
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341Ý
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296ì
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493â
flatten_23/PartitionedCallPartitionedCall+dropout_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357ù
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0
fc1_617674
fc1_617676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370
"dropout_71/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_70/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_71/StatefulPartitionedCall:output:0
fc2_617680
fc2_617682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall#^dropout_71/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2H
"dropout_71/StatefulPartitionedCall"dropout_71/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24
Þ
Ò
7__inference_batch_normalization_47_layer_call_fn_617940

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þ%
ë
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ê
(__inference_Predict_layer_call_fn_617742

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs


ê
(__inference_Predict_layer_call_fn_617767

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¯

e
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Â
b
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
õ
E
)__inference_MaxPool2_layer_call_fn_618022

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
à

,__inference_layer_conv2_layer_call_fn_617912

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Î&
è
C__inference_Predict_layer_call_and_return_conditional_losses_617653
input_24(
layer_conv2_617623:  
layer_conv2_617625: +
batch_normalization_47_617628: +
batch_normalization_47_617630: +
batch_normalization_47_617632: +
batch_normalization_47_617634: 

fc1_617641:	(@

fc1_617643:@

fc2_617647:@

fc2_617649:
identity¢.batch_normalization_47/StatefulPartitionedCall¢fc1/StatefulPartitionedCall¢fc2/StatefulPartitionedCall¢#layer_conv2/StatefulPartitionedCall
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_24layer_conv2_617623layer_conv2_617625*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617628batch_normalization_47_617630batch_normalization_47_617632batch_normalization_47_617634*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226ø
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341Ý
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296Ü
dropout_70/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349Ú
flatten_23/PartitionedCallPartitionedCall#dropout_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357ù
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0
fc1_617641
fc1_617643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370Ú
dropout_71/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381ù
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_71/PartitionedCall:output:0
fc2_617647
fc2_617649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
input_24
¬r
û
"__inference__traced_restore_618324
file_prefix9
#assignvariableop_layer_conv2_kernel: 1
#assignvariableop_1_layer_conv2_bias: =
/assignvariableop_2_batch_normalization_47_gamma: <
.assignvariableop_3_batch_normalization_47_beta: C
5assignvariableop_4_batch_normalization_47_moving_mean: G
9assignvariableop_5_batch_normalization_47_moving_variance: 0
assignvariableop_6_fc1_kernel:	(@)
assignvariableop_7_fc1_bias:@/
assignvariableop_8_fc2_kernel:@)
assignvariableop_9_fc2_bias:#
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: &
assignvariableop_12_momentum: &
assignvariableop_13_sgd_iter:	 %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: I
3assignvariableop_18_sgd_layer_conv2_kernel_momentum: ?
1assignvariableop_19_sgd_layer_conv2_bias_momentum: K
=assignvariableop_20_sgd_batch_normalization_47_gamma_momentum: J
<assignvariableop_21_sgd_batch_normalization_47_beta_momentum: >
+assignvariableop_22_sgd_fc1_kernel_momentum:	(@7
)assignvariableop_23_sgd_fc1_bias_momentum:@=
+assignvariableop_24_sgd_fc2_kernel_momentum:@7
)assignvariableop_25_sgd_fc2_bias_momentum:
identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9£
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*É
value¿B¼B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¦
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ¦
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOpAssignVariableOp#assignvariableop_layer_conv2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_1AssignVariableOp#assignvariableop_1_layer_conv2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_47_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_47_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_47_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_47_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_8AssignVariableOpassignvariableop_8_fc2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_9AssignVariableOpassignvariableop_9_fc2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_12AssignVariableOpassignvariableop_12_momentumIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:µ
AssignVariableOp_13AssignVariableOpassignvariableop_13_sgd_iterIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_18AssignVariableOp3assignvariableop_18_sgd_layer_conv2_kernel_momentumIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_19AssignVariableOp1assignvariableop_19_sgd_layer_conv2_bias_momentumIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ö
AssignVariableOp_20AssignVariableOp=assignvariableop_20_sgd_batch_normalization_47_gamma_momentumIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_21AssignVariableOp<assignvariableop_21_sgd_batch_normalization_47_beta_momentumIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_22AssignVariableOp+assignvariableop_22_sgd_fc1_kernel_momentumIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_23AssignVariableOp)assignvariableop_23_sgd_fc1_bias_momentumIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_24AssignVariableOp+assignvariableop_24_sgd_fc2_kernel_momentumIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_25AssignVariableOp)assignvariableop_25_sgd_fc2_bias_momentumIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ø
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
È&
æ
C__inference_Predict_layer_call_and_return_conditional_losses_617401

inputs(
layer_conv2_617322:  
layer_conv2_617324: +
batch_normalization_47_617327: +
batch_normalization_47_617329: +
batch_normalization_47_617331: +
batch_normalization_47_617333: 

fc1_617371:	(@

fc1_617373:@

fc2_617395:@

fc2_617397:
identity¢.batch_normalization_47/StatefulPartitionedCall¢fc1/StatefulPartitionedCall¢fc2/StatefulPartitionedCall¢#layer_conv2/StatefulPartitionedCall
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_617322layer_conv2_617324*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617327batch_normalization_47_617329batch_normalization_47_617331batch_normalization_47_617333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226ø
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341Ý
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296Ü
dropout_70/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349Ú
flatten_23/PartitionedCallPartitionedCall#dropout_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357ù
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0
fc1_617371
fc1_617373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370Ú
dropout_71/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381ù
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_71/PartitionedCall:output:0
fc2_617395
fc2_617397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs


e
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸

$__inference_fc2_layer_call_fn_618124

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
+__inference_dropout_70_layer_call_fn_618040

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
µ
G
+__inference_dropout_70_layer_call_fn_618035

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
»
J
.__inference_activation_47_layer_call_fn_618012

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
 
_user_specified_nameinputs
ù

G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¯

e
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
B
input_246
serving_default_input_24:0ÿÿÿÿÿÿÿÿÿÀ7
fc20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¹ñ
Í
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
ê
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#axis
	$gamma
%beta
&moving_mean
'moving_variance"
_tf_keras_layer
¥
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_random_generator"
_tf_keras_layer
¥
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
»
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
¼
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator"
_tf_keras_layer
»
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
f
0
1
$2
%3
&4
'5
G6
H7
V8
W9"
trackable_list_wrapper
X
0
1
$2
%3
G4
H5
V6
W7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Õ
]trace_0
^trace_1
_trace_2
`trace_32ê
(__inference_Predict_layer_call_fn_617424
(__inference_Predict_layer_call_fn_617742
(__inference_Predict_layer_call_fn_617767
(__inference_Predict_layer_call_fn_617620¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z]trace_0z^trace_1z_trace_2z`trace_3
Á
atrace_0
btrace_1
ctrace_2
dtrace_32Ö
C__inference_Predict_layer_call_and_return_conditional_losses_617821
C__inference_Predict_layer_call_and_return_conditional_losses_617903
C__inference_Predict_layer_call_and_return_conditional_losses_617653
C__inference_Predict_layer_call_and_return_conditional_losses_617686¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0zbtrace_1zctrace_2zdtrace_3
ÍBÊ
!__inference__wrapped_model_617202input_24"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ
	edecay
flearning_rate
gmomentum
hitermomentumºmomentum»$momentum¼%momentum½Gmomentum¾Hmomentum¿VmomentumÀWmomentumÁ"
	optimizer
,
iserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
otrace_02Ó
,__inference_layer_conv2_layer_call_fn_617912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0

ptrace_02î
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0
(:& 2layer_conv2/kernel
: 2layer_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
ß
vtrace_0
wtrace_12¨
7__inference_batch_normalization_47_layer_call_fn_617940
7__inference_batch_normalization_47_layer_call_fn_617953³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0zwtrace_1

xtrace_0
ytrace_12Þ
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zxtrace_0zytrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_47/gamma
):' 2batch_normalization_47/beta
2:0  (2"batch_normalization_47/moving_mean
6:4  (2&batch_normalization_47/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Õ
.__inference_activation_47_layer_call_fn_618012¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ð
I__inference_activation_47_layer_call_and_return_conditional_losses_618017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_MaxPool2_layer_call_fn_618022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
Ë
trace_0
trace_12
+__inference_dropout_70_layer_call_fn_618035
+__inference_dropout_70_layer_call_fn_618040³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Æ
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_flatten_23_layer_call_fn_618062¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ê
trace_02Ë
$__inference_fc1_layer_call_fn_618077¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02æ
?__inference_fc1_layer_call_and_return_conditional_losses_618088¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
:	(@2
fc1/kernel
:@2fc1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Ë
¤trace_0
¥trace_12
+__inference_dropout_71_layer_call_fn_618093
+__inference_dropout_71_layer_call_fn_618098³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¤trace_0z¥trace_1

¦trace_0
§trace_12Æ
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0z§trace_1
"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ê
­trace_02Ë
$__inference_fc2_layer_call_fn_618124¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0

®trace_02æ
?__inference_fc2_layer_call_and_return_conditional_losses_618135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0
:@2
fc2/kernel
:2fc2/bias
.
&0
'1"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
¯0
°1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
(__inference_Predict_layer_call_fn_617424input_24"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
(__inference_Predict_layer_call_fn_617742inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
(__inference_Predict_layer_call_fn_617767inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
(__inference_Predict_layer_call_fn_617620input_24"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_Predict_layer_call_and_return_conditional_losses_617821inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_Predict_layer_call_and_return_conditional_losses_617903inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_Predict_layer_call_and_return_conditional_losses_617653input_24"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_Predict_layer_call_and_return_conditional_losses_617686input_24"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
ÌBÉ
$__inference_signature_wrapper_617717input_24"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_layer_conv2_layer_call_fn_617912inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
7__inference_batch_normalization_47_layer_call_fn_617940inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
7__inference_batch_normalization_47_layer_call_fn_617953inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_activation_47_layer_call_fn_618012inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_activation_47_layer_call_and_return_conditional_losses_618017inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_MaxPool2_layer_call_fn_618022inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ðBí
+__inference_dropout_70_layer_call_fn_618035inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ðBí
+__inference_dropout_70_layer_call_fn_618040inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_flatten_23_layer_call_fn_618062inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ØBÕ
$__inference_fc1_layer_call_fn_618077inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
óBð
?__inference_fc1_layer_call_and_return_conditional_losses_618088inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ðBí
+__inference_dropout_71_layer_call_fn_618093inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ðBí
+__inference_dropout_71_layer_call_fn_618098inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ØBÕ
$__inference_fc2_layer_call_fn_618124inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
óBð
?__inference_fc2_layer_call_and_return_conditional_losses_618135inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
±	variables
²	keras_api

³total

´count"
_tf_keras_metric
c
µ	variables
¶	keras_api

·total

¸count
¹
_fn_kwargs"
_tf_keras_metric
0
³0
´1"
trackable_list_wrapper
.
±	variables"
_generic_user_object
:  (2total
:  (2count
0
·0
¸1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
3:1 2SGD/layer_conv2/kernel/momentum
):' 2SGD/layer_conv2/bias/momentum
5:3 2)SGD/batch_normalization_47/gamma/momentum
4:2 2(SGD/batch_normalization_47/beta/momentum
(:&	(@2SGD/fc1/kernel/momentum
!:@2SGD/fc1/bias/momentum
':%@2SGD/fc2/kernel/momentum
!:2SGD/fc2/bias/momentumÔ
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "B¢?
85
tensor_0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
)__inference_MaxPool2_layer_call_fn_618022E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "74
unknown'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
C__inference_Predict_layer_call_and_return_conditional_losses_617653z
'$&%GHVW>¢;
4¢1
'$
input_24ÿÿÿÿÿÿÿÿÿÀ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Á
C__inference_Predict_layer_call_and_return_conditional_losses_617686z
&'$%GHVW>¢;
4¢1
'$
input_24ÿÿÿÿÿÿÿÿÿÀ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¿
C__inference_Predict_layer_call_and_return_conditional_losses_617821x
'$&%GHVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÀ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¿
C__inference_Predict_layer_call_and_return_conditional_losses_617903x
&'$%GHVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÀ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_Predict_layer_call_fn_617424o
'$&%GHVW>¢;
4¢1
'$
input_24ÿÿÿÿÿÿÿÿÿÀ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
(__inference_Predict_layer_call_fn_617620o
&'$%GHVW>¢;
4¢1
'$
input_24ÿÿÿÿÿÿÿÿÿÀ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
(__inference_Predict_layer_call_fn_617742m
'$&%GHVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÀ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
(__inference_Predict_layer_call_fn_617767m
&'$%GHVW<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÀ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
!__inference__wrapped_model_617202o
'$&%GHVW6¢3
,¢)
'$
input_24ÿÿÿÿÿÿÿÿÿÀ
ª ")ª&
$
fc2
fc2ÿÿÿÿÿÿÿÿÿ¶
I__inference_activation_47_layer_call_and_return_conditional_losses_618017i4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÀ 
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿÀ 
 
.__inference_activation_47_layer_call_fn_618012^4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÀ 
ª "&#
unknownÿÿÿÿÿÿÿÿÿÀ Ú
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973'$&%@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "9¢6
/,
tensor_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ú
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007&'$%@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "9¢6
/,
tensor_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ³
7__inference_batch_normalization_47_layer_call_fn_617940x'$&%@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª ".+
unknownÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ³
7__inference_batch_normalization_47_layer_call_fn_617953x&'$%@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª ".+
unknownÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ·
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045m8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿ  
 ·
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057m8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿ  
 
+__inference_dropout_70_layer_call_fn_618035b8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "&#
unknownÿÿÿÿÿÿÿÿÿ  
+__inference_dropout_70_layer_call_fn_618040b8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "&#
unknownÿÿÿÿÿÿÿÿÿ  ­
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103c3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ@
 ­
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115c3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_dropout_71_layer_call_fn_618093X3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "!
unknownÿÿÿÿÿÿÿÿÿ@
+__inference_dropout_71_layer_call_fn_618098X3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "!
unknownÿÿÿÿÿÿÿÿÿ@§
?__inference_fc1_layer_call_and_return_conditional_losses_618088dGH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ(
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ@
 
$__inference_fc1_layer_call_fn_618077YGH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ(
ª "!
unknownÿÿÿÿÿÿÿÿÿ@¦
?__inference_fc2_layer_call_and_return_conditional_losses_618135cVW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
$__inference_fc2_layer_call_fn_618124XVW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "!
unknownÿÿÿÿÿÿÿÿÿ¯
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068e4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ(
 
+__inference_flatten_23_layer_call_fn_618062Z4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ  
ª ""
unknownÿÿÿÿÿÿÿÿÿ(¸
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927m4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÀ
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿÀ 
 
,__inference_layer_conv2_layer_call_fn_617912b4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÀ
ª "&#
unknownÿÿÿÿÿÿÿÿÿÀ £
$__inference_signature_wrapper_617717{
'$&%GHVWB¢?
¢ 
8ª5
3
input_24'$
input_24ÿÿÿÿÿÿÿÿÿÀ")ª&
$
fc2
fc2ÿÿÿÿÿÿÿÿÿ