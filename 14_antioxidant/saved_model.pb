Кё
╬Ю
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
Џ
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
ѓ
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
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
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ощ
ѓ
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
і
SGD/fc2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameSGD/fc2/kernel/momentum
Ѓ
+SGD/fc2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc2/kernel/momentum*
_output_shapes

:@*
dtype0
ѓ
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
І
SGD/fc1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ(@*(
shared_nameSGD/fc1/kernel/momentum
ё
+SGD/fc1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc1/kernel/momentum*
_output_shapes
:	ђ(@*
dtype0
е
(SGD/batch_normalization_47/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_47/beta/momentum
А
<SGD/batch_normalization_47/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_47/beta/momentum*
_output_shapes
: *
dtype0
ф
)SGD/batch_normalization_47/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_47/gamma/momentum
Б
=SGD/batch_normalization_47/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_47/gamma/momentum*
_output_shapes
: *
dtype0
њ
SGD/layer_conv2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/layer_conv2/bias/momentum
І
1SGD/layer_conv2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer_conv2/bias/momentum*
_output_shapes
: *
dtype0
ъ
SGD/layer_conv2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!SGD/layer_conv2/kernel/momentum
Ќ
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
shape:	ђ(@*
shared_name
fc1/kernel
j
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
_output_shapes
:	ђ(@*
dtype0
ц
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_47/moving_variance
Ю
:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes
: *
dtype0
ю
"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_47/moving_mean
Ћ
6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes
: *
dtype0
ј
batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_47/beta
Є
/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes
: *
dtype0
љ
batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_47/gamma
Ѕ
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
ё
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
Ё
serving_default_input_24Placeholder*,
_output_shapes
:         └*
dtype0*!
shape:         └
а
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
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_617717

NoOpNoOp
зJ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«J
valueцJBАJ BџJ
Х
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
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
Н
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
ј
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
ј
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
Ц
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_random_generator* 
ј
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
д
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias*
Ц
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator* 
д
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
░
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
┬
	edecay
flearning_rate
gmomentum
hitermomentum║momentum╗$momentum╝%momentumйGmomentumЙHmomentum┐Vmomentum└Wmomentum┴*

iserving_default* 

0
1*

0
1*
* 
Њ
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
Њ
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
Љ
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
ђtrace_0* 
* 
* 
* 
ќ
Ђnon_trainable_variables
ѓlayers
Ѓmetrics
 ёlayer_regularization_losses
Ёlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

єtrace_0* 

Єtrace_0* 
* 
* 
* 
ќ
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

Їtrace_0
јtrace_1* 

Јtrace_0
љtrace_1* 
* 
* 
* 
* 
ќ
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

ќtrace_0* 

Ќtrace_0* 

G0
H1*

G0
H1*
* 
ў
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

Юtrace_0* 

ъtrace_0* 
ZT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

цtrace_0
Цtrace_1* 

дtrace_0
Дtrace_1* 
* 

V0
W1*

V0
W1*
* 
ў
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

Гtrace_0* 

«trace_0* 
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
»0
░1*
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
▒	variables
▓	keras_api

│total

┤count*
M
х	variables
Х	keras_api

иtotal

Иcount
╣
_fn_kwargs*

│0
┤1*

▒	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

и0
И1*

х	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Њї
VARIABLE_VALUESGD/layer_conv2/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUESGD/layer_conv2/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
юЋ
VARIABLE_VALUE)SGD/batch_normalization_47/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
џЊ
VARIABLE_VALUE(SGD/batch_normalization_47/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUESGD/fc1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Єђ
VARIABLE_VALUESGD/fc1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUESGD/fc2/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Єђ
VARIABLE_VALUESGD/fc2/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Н

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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_618236
╚
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_618324вС
џ

ы
?__inference_fc1_layer_call_and_return_conditional_losses_617370

inputs1
matmul_readvariableop_resource:	ђ(@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ(
 
_user_specified_nameinputs
Џ

В
(__inference_Predict_layer_call_fn_617620
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	ђ(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         └
"
_user_specified_name
input_24
┘
d
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ь
d
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         а `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         а "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
џ

ы
?__inference_fc1_layer_call_and_return_conditional_losses_618088

inputs1
matmul_readvariableop_resource:	ђ(@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ(
 
_user_specified_nameinputs
э	
У
$__inference_signature_wrapper_617717
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	ђ(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_617202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         └
"
_user_specified_name
input_24
ь
d
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         а `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         а "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
▄
м
7__inference_batch_normalization_47_layer_call_fn_617953

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Џ

­
?__inference_fc2_layer_call_and_return_conditional_losses_617394

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
щ
ќ
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ѓ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └ *
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         └ *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         └ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         └ ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
Г
G
+__inference_flatten_23_layer_call_fn_618062

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
┘
d
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
з
d
+__inference_dropout_71_layer_call_fn_618098

inputs
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
А
G
+__inference_dropout_71_layer_call_fn_618093

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
АC
 
C__inference_Predict_layer_call_and_return_conditional_losses_617821

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: F
8batch_normalization_47_batchnorm_readvariableop_resource: J
<batch_normalization_47_batchnorm_mul_readvariableop_resource: H
:batch_normalization_47_batchnorm_readvariableop_1_resource: H
:batch_normalization_47_batchnorm_readvariableop_2_resource: 5
"fc1_matmul_readvariableop_resource:	ђ(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identityѕб/batch_normalization_47/batchnorm/ReadVariableOpб1batch_normalization_47/batchnorm/ReadVariableOp_1б1batch_normalization_47/batchnorm/ReadVariableOp_2б3batch_normalization_47/batchnorm/mul/ReadVariableOpбfc1/BiasAdd/ReadVariableOpбfc1/MatMul/ReadVariableOpбfc2/BiasAdd/ReadVariableOpбfc2/MatMul/ReadVariableOpб"layer_conv2/BiasAdd/ReadVariableOpб.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpl
!layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        џ
layer_conv2/Conv1D/ExpandDims
ExpandDimsinputs*layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └ф
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0e
#layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ─
layer_conv2/Conv1D/ExpandDims_1
ExpandDims6layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Л
layer_conv2/Conv1DConv2D&layer_conv2/Conv1D/ExpandDims:output:0(layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └ *
paddingSAME*
strides
Ў
layer_conv2/Conv1D/SqueezeSqueezelayer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         └ *
squeeze_dims

§        і
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0д
layer_conv2/BiasAddBiasAdd#layer_conv2/Conv1D/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         └ ц
/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0k
&batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╝
$batch_normalization_47/batchnorm/addAddV27batch_normalization_47/batchnorm/ReadVariableOp:value:0/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_47/batchnorm/RsqrtRsqrt(batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: г
3batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0╣
$batch_normalization_47/batchnorm/mulMul*batch_normalization_47/batchnorm/Rsqrt:y:0;batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: г
&batch_normalization_47/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └ е
1batch_normalization_47/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_47_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0и
&batch_normalization_47/batchnorm/mul_2Mul9batch_normalization_47/batchnorm/ReadVariableOp_1:value:0(batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: е
1batch_normalization_47/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_47_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0и
$batch_normalization_47/batchnorm/subSub9batch_normalization_47/batchnorm/ReadVariableOp_2:value:0*batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ╝
&batch_normalization_47/batchnorm/add_1AddV2*batch_normalization_47/batchnorm/mul_1:z:0(batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └ }
activation_47/ReluRelu*batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └ Y
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :а
MaxPool2/ExpandDims
ExpandDims activation_47/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └ д
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:         а *
ksize
*
paddingSAME*
strides
ё
MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:         а *
squeeze_dims
q
dropout_70/IdentityIdentityMaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:         а a
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ѕ
flatten_23/ReshapeReshapedropout_70/Identity:output:0flatten_23/Const:output:0*
T0*(
_output_shapes
:         ђ(}
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	ђ(@*
dtype0є

fc1/MatMulMatMulflatten_23/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @X
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:         @i
dropout_71/IdentityIdentityfc1/Relu:activations:0*
T0*'
_output_shapes
:         @|
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Є

fc2/MatMulMatMuldropout_71/Identity:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ^
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:         d
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp0^batch_normalization_47/batchnorm/ReadVariableOp2^batch_normalization_47/batchnorm/ReadVariableOp_12^batch_normalization_47/batchnorm/ReadVariableOp_24^batch_normalization_47/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2b
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
:         └
 
_user_specified_nameinputs
╗
њ
$__inference_fc1_layer_call_fn_618077

inputs
unknown:	ђ(@
	unknown_0:@
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ(
 
_user_specified_nameinputs
р
e
I__inference_activation_47_layer_call_and_return_conditional_losses_618017

inputs
identityK
ReluReluinputs*
T0*,
_output_shapes
:         └ _
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         └ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └ :T P
,
_output_shapes
:         └ 
 
_user_specified_nameinputs
╚
`
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
┬
b
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
Ю

В
(__inference_Predict_layer_call_fn_617424
input_24
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	ђ(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         └
"
_user_specified_name
input_24
ЗK
 	
!__inference__wrapped_model_617202
input_24U
?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource: A
3predict_layer_conv2_biasadd_readvariableop_resource: N
@predict_batch_normalization_47_batchnorm_readvariableop_resource: R
Dpredict_batch_normalization_47_batchnorm_mul_readvariableop_resource: P
Bpredict_batch_normalization_47_batchnorm_readvariableop_1_resource: P
Bpredict_batch_normalization_47_batchnorm_readvariableop_2_resource: =
*predict_fc1_matmul_readvariableop_resource:	ђ(@9
+predict_fc1_biasadd_readvariableop_resource:@<
*predict_fc2_matmul_readvariableop_resource:@9
+predict_fc2_biasadd_readvariableop_resource:
identityѕб7Predict/batch_normalization_47/batchnorm/ReadVariableOpб9Predict/batch_normalization_47/batchnorm/ReadVariableOp_1б9Predict/batch_normalization_47/batchnorm/ReadVariableOp_2б;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOpб"Predict/fc1/BiasAdd/ReadVariableOpб!Predict/fc1/MatMul/ReadVariableOpб"Predict/fc2/BiasAdd/ReadVariableOpб!Predict/fc2/MatMul/ReadVariableOpб*Predict/layer_conv2/BiasAdd/ReadVariableOpб6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpt
)Predict/layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        г
%Predict/layer_conv2/Conv1D/ExpandDims
ExpandDimsinput_242Predict/layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └║
6Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0m
+Predict/layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ▄
'Predict/layer_conv2/Conv1D/ExpandDims_1
ExpandDims>Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:04Predict/layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ж
Predict/layer_conv2/Conv1DConv2D.Predict/layer_conv2/Conv1D/ExpandDims:output:00Predict/layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └ *
paddingSAME*
strides
Е
"Predict/layer_conv2/Conv1D/SqueezeSqueeze#Predict/layer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         └ *
squeeze_dims

§        џ
*Predict/layer_conv2/BiasAdd/ReadVariableOpReadVariableOp3predict_layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Й
Predict/layer_conv2/BiasAddBiasAdd+Predict/layer_conv2/Conv1D/Squeeze:output:02Predict/layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         └ ┤
7Predict/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp@predict_batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0s
.Predict/batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:н
,Predict/batch_normalization_47/batchnorm/addAddV2?Predict/batch_normalization_47/batchnorm/ReadVariableOp:value:07Predict/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: ј
.Predict/batch_normalization_47/batchnorm/RsqrtRsqrt0Predict/batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: ╝
;Predict/batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOpDpredict_batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Л
,Predict/batch_normalization_47/batchnorm/mulMul2Predict/batch_normalization_47/batchnorm/Rsqrt:y:0CPredict/batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: ─
.Predict/batch_normalization_47/batchnorm/mul_1Mul$Predict/layer_conv2/BiasAdd:output:00Predict/batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └ И
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_1ReadVariableOpBpredict_batch_normalization_47_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0¤
.Predict/batch_normalization_47/batchnorm/mul_2MulAPredict/batch_normalization_47/batchnorm/ReadVariableOp_1:value:00Predict/batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: И
9Predict/batch_normalization_47/batchnorm/ReadVariableOp_2ReadVariableOpBpredict_batch_normalization_47_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0¤
,Predict/batch_normalization_47/batchnorm/subSubAPredict/batch_normalization_47/batchnorm/ReadVariableOp_2:value:02Predict/batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: н
.Predict/batch_normalization_47/batchnorm/add_1AddV22Predict/batch_normalization_47/batchnorm/mul_1:z:00Predict/batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └ Ї
Predict/activation_47/ReluRelu2Predict/batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └ a
Predict/MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :И
Predict/MaxPool2/ExpandDims
ExpandDims(Predict/activation_47/Relu:activations:0(Predict/MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └ Х
Predict/MaxPool2/MaxPoolMaxPool$Predict/MaxPool2/ExpandDims:output:0*0
_output_shapes
:         а *
ksize
*
paddingSAME*
strides
ћ
Predict/MaxPool2/SqueezeSqueeze!Predict/MaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:         а *
squeeze_dims
Ђ
Predict/dropout_70/IdentityIdentity!Predict/MaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:         а i
Predict/flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"       А
Predict/flatten_23/ReshapeReshape$Predict/dropout_70/Identity:output:0!Predict/flatten_23/Const:output:0*
T0*(
_output_shapes
:         ђ(Ї
!Predict/fc1/MatMul/ReadVariableOpReadVariableOp*predict_fc1_matmul_readvariableop_resource*
_output_shapes
:	ђ(@*
dtype0ъ
Predict/fc1/MatMulMatMul#Predict/flatten_23/Reshape:output:0)Predict/fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @і
"Predict/fc1/BiasAdd/ReadVariableOpReadVariableOp+predict_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0џ
Predict/fc1/BiasAddBiasAddPredict/fc1/MatMul:product:0*Predict/fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @h
Predict/fc1/ReluReluPredict/fc1/BiasAdd:output:0*
T0*'
_output_shapes
:         @y
Predict/dropout_71/IdentityIdentityPredict/fc1/Relu:activations:0*
T0*'
_output_shapes
:         @ї
!Predict/fc2/MatMul/ReadVariableOpReadVariableOp*predict_fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ъ
Predict/fc2/MatMulMatMul$Predict/dropout_71/Identity:output:0)Predict/fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         і
"Predict/fc2/BiasAdd/ReadVariableOpReadVariableOp+predict_fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0џ
Predict/fc2/BiasAddBiasAddPredict/fc2/MatMul:product:0*Predict/fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         n
Predict/fc2/SoftmaxSoftmaxPredict/fc2/BiasAdd:output:0*
T0*'
_output_shapes
:         l
IdentityIdentityPredict/fc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         «
NoOpNoOp8^Predict/batch_normalization_47/batchnorm/ReadVariableOp:^Predict/batch_normalization_47/batchnorm/ReadVariableOp_1:^Predict/batch_normalization_47/batchnorm/ReadVariableOp_2<^Predict/batch_normalization_47/batchnorm/mul/ReadVariableOp#^Predict/fc1/BiasAdd/ReadVariableOp"^Predict/fc1/MatMul/ReadVariableOp#^Predict/fc2/BiasAdd/ReadVariableOp"^Predict/fc2/MatMul/ReadVariableOp+^Predict/layer_conv2/BiasAdd/ReadVariableOp7^Predict/layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2r
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
:         └
"
_user_specified_name
input_24
ї

e
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
р
e
I__inference_activation_47_layer_call_and_return_conditional_losses_617341

inputs
identityK
ReluReluinputs*
T0*,
_output_shapes
:         └ _
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         └ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └ :T P
,
_output_shapes
:         └ 
 
_user_specified_nameinputs
├)
░
C__inference_Predict_layer_call_and_return_conditional_losses_617572

inputs(
layer_conv2_617542:  
layer_conv2_617544: +
batch_normalization_47_617547: +
batch_normalization_47_617549: +
batch_normalization_47_617551: +
batch_normalization_47_617553: 

fc1_617560:	ђ(@

fc1_617562:@

fc2_617566:@

fc2_617568:
identityѕб.batch_normalization_47/StatefulPartitionedCallб"dropout_70/StatefulPartitionedCallб"dropout_71/StatefulPartitionedCallбfc1/StatefulPartitionedCallбfc2/StatefulPartitionedCallб#layer_conv2/StatefulPartitionedCallЂ
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_617542layer_conv2_617544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321Њ
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617547batch_normalization_47_617549batch_normalization_47_617551batch_normalization_47_617553*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273Э
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341П
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296В
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493Р
flatten_23/PartitionedCallPartitionedCall+dropout_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357щ
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
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370Ј
"dropout_71/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_70/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454Ђ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall#^dropout_71/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2H
"dropout_71/StatefulPartitionedCall"dropout_71/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
╚
`
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           Ц
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
Ѓ
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
љ
▒
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
 :                   z
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
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
■%
в
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
 :                   h
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
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Џm
у	
C__inference_Predict_layer_call_and_return_conditional_losses_617903

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: L
>batch_normalization_47_assignmovingavg_readvariableop_resource: N
@batch_normalization_47_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_47_batchnorm_mul_readvariableop_resource: F
8batch_normalization_47_batchnorm_readvariableop_resource: 5
"fc1_matmul_readvariableop_resource:	ђ(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identityѕб&batch_normalization_47/AssignMovingAvgб5batch_normalization_47/AssignMovingAvg/ReadVariableOpб(batch_normalization_47/AssignMovingAvg_1б7batch_normalization_47/AssignMovingAvg_1/ReadVariableOpб/batch_normalization_47/batchnorm/ReadVariableOpб3batch_normalization_47/batchnorm/mul/ReadVariableOpбfc1/BiasAdd/ReadVariableOpбfc1/MatMul/ReadVariableOpбfc2/BiasAdd/ReadVariableOpбfc2/MatMul/ReadVariableOpб"layer_conv2/BiasAdd/ReadVariableOpб.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpl
!layer_conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        џ
layer_conv2/Conv1D/ExpandDims
ExpandDimsinputs*layer_conv2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └ф
.layer_conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0e
#layer_conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ─
layer_conv2/Conv1D/ExpandDims_1
ExpandDims6layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Л
layer_conv2/Conv1DConv2D&layer_conv2/Conv1D/ExpandDims:output:0(layer_conv2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └ *
paddingSAME*
strides
Ў
layer_conv2/Conv1D/SqueezeSqueezelayer_conv2/Conv1D:output:0*
T0*,
_output_shapes
:         └ *
squeeze_dims

§        і
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0д
layer_conv2/BiasAddBiasAdd#layer_conv2/Conv1D/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         └ є
5batch_normalization_47/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       К
#batch_normalization_47/moments/meanMeanlayer_conv2/BiasAdd:output:0>batch_normalization_47/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(ќ
+batch_normalization_47/moments/StopGradientStopGradient,batch_normalization_47/moments/mean:output:0*
T0*"
_output_shapes
: л
0batch_normalization_47/moments/SquaredDifferenceSquaredDifferencelayer_conv2/BiasAdd:output:04batch_normalization_47/moments/StopGradient:output:0*
T0*,
_output_shapes
:         └ і
9batch_normalization_47/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       у
'batch_normalization_47/moments/varianceMean4batch_normalization_47/moments/SquaredDifference:z:0Bbatch_normalization_47/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(ю
&batch_normalization_47/moments/SqueezeSqueeze,batch_normalization_47/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 б
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
О#<░
5batch_normalization_47/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_47_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0к
*batch_normalization_47/AssignMovingAvg/subSub=batch_normalization_47/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_47/moments/Squeeze:output:0*
T0*
_output_shapes
: й
*batch_normalization_47/AssignMovingAvg/mulMul.batch_normalization_47/AssignMovingAvg/sub:z:05batch_normalization_47/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ѕ
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
О#<┤
7batch_normalization_47/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_47_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0╠
,batch_normalization_47/AssignMovingAvg_1/subSub?batch_normalization_47/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_47/moments/Squeeze_1:output:0*
T0*
_output_shapes
: ├
,batch_normalization_47/AssignMovingAvg_1/mulMul0batch_normalization_47/AssignMovingAvg_1/sub:z:07batch_normalization_47/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: љ
(batch_normalization_47/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_47_assignmovingavg_1_readvariableop_resource0batch_normalization_47/AssignMovingAvg_1/mul:z:08^batch_normalization_47/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_47/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Х
$batch_normalization_47/batchnorm/addAddV21batch_normalization_47/moments/Squeeze_1:output:0/batch_normalization_47/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&batch_normalization_47/batchnorm/RsqrtRsqrt(batch_normalization_47/batchnorm/add:z:0*
T0*
_output_shapes
: г
3batch_normalization_47/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_47_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0╣
$batch_normalization_47/batchnorm/mulMul*batch_normalization_47/batchnorm/Rsqrt:y:0;batch_normalization_47/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: г
&batch_normalization_47/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └ Г
&batch_normalization_47/batchnorm/mul_2Mul/batch_normalization_47/moments/Squeeze:output:0(batch_normalization_47/batchnorm/mul:z:0*
T0*
_output_shapes
: ц
/batch_normalization_47/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_47_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0х
$batch_normalization_47/batchnorm/subSub7batch_normalization_47/batchnorm/ReadVariableOp:value:0*batch_normalization_47/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ╝
&batch_normalization_47/batchnorm/add_1AddV2*batch_normalization_47/batchnorm/mul_1:z:0(batch_normalization_47/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └ }
activation_47/ReluRelu*batch_normalization_47/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └ Y
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :а
MaxPool2/ExpandDims
ExpandDims activation_47/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └ д
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:         а *
ksize
*
paddingSAME*
strides
ё
MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:         а *
squeeze_dims
]
dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?њ
dropout_70/dropout/MulMulMaxPool2/Squeeze:output:0!dropout_70/dropout/Const:output:0*
T0*,
_output_shapes
:         а a
dropout_70/dropout/ShapeShapeMaxPool2/Squeeze:output:0*
T0*
_output_shapes
:Д
/dropout_70/dropout/random_uniform/RandomUniformRandomUniform!dropout_70/dropout/Shape:output:0*
T0*,
_output_shapes
:         а *
dtype0f
!dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>╠
dropout_70/dropout/GreaterEqualGreaterEqual8dropout_70/dropout/random_uniform/RandomUniform:output:0*dropout_70/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         а _
dropout_70/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
dropout_70/dropout/SelectV2SelectV2#dropout_70/dropout/GreaterEqual:z:0dropout_70/dropout/Mul:z:0#dropout_70/dropout/Const_1:output:0*
T0*,
_output_shapes
:         а a
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Љ
flatten_23/ReshapeReshape$dropout_70/dropout/SelectV2:output:0flatten_23/Const:output:0*
T0*(
_output_shapes
:         ђ(}
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	ђ(@*
dtype0є

fc1/MatMulMatMulflatten_23/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @X
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:         @]
dropout_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?і
dropout_71/dropout/MulMulfc1/Relu:activations:0!dropout_71/dropout/Const:output:0*
T0*'
_output_shapes
:         @^
dropout_71/dropout/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:б
/dropout_71/dropout/random_uniform/RandomUniformRandomUniform!dropout_71/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0f
!dropout_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>К
dropout_71/dropout/GreaterEqualGreaterEqual8dropout_71/dropout/random_uniform/RandomUniform:output:0*dropout_71/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @_
dropout_71/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_71/dropout/SelectV2SelectV2#dropout_71/dropout/GreaterEqual:z:0dropout_71/dropout/Mul:z:0#dropout_71/dropout/Const_1:output:0*
T0*'
_output_shapes
:         @|
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј

fc2/MatMulMatMul$dropout_71/dropout/SelectV2:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ^
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:         d
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOp'^batch_normalization_47/AssignMovingAvg6^batch_normalization_47/AssignMovingAvg/ReadVariableOp)^batch_normalization_47/AssignMovingAvg_18^batch_normalization_47/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_47/batchnorm/ReadVariableOp4^batch_normalization_47/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2P
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
:         └
 
_user_specified_nameinputs
Џ

­
?__inference_fc2_layer_call_and_return_conditional_losses_618135

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ы;
ф
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

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: а
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╔
value┐B╝B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ╚
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_layer_conv2_kernel_read_readvariableop+savev2_layer_conv2_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_sgd_layer_conv2_kernel_momentum_read_readvariableop8savev2_sgd_layer_conv2_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_47_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_47_beta_momentum_read_readvariableop2savev2_sgd_fc1_kernel_momentum_read_readvariableop0savev2_sgd_fc1_bias_momentum_read_readvariableop2savev2_sgd_fc2_kernel_momentum_read_readvariableop0savev2_sgd_fc2_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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

identity_1Identity_1:output:0*и
_input_shapesЦ
б: : : : : : : :	ђ(@:@:@:: : : : : : : : : : : : :	ђ(@:@:@:: 2(
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
:	ђ(@: 
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
:	ђ(@: 
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
╔)
▓
C__inference_Predict_layer_call_and_return_conditional_losses_617686
input_24(
layer_conv2_617656:  
layer_conv2_617658: +
batch_normalization_47_617661: +
batch_normalization_47_617663: +
batch_normalization_47_617665: +
batch_normalization_47_617667: 

fc1_617674:	ђ(@

fc1_617676:@

fc2_617680:@

fc2_617682:
identityѕб.batch_normalization_47/StatefulPartitionedCallб"dropout_70/StatefulPartitionedCallб"dropout_71/StatefulPartitionedCallбfc1/StatefulPartitionedCallбfc2/StatefulPartitionedCallб#layer_conv2/StatefulPartitionedCallЃ
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_24layer_conv2_617656layer_conv2_617658*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321Њ
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617661batch_normalization_47_617663batch_normalization_47_617665batch_normalization_47_617667*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273Э
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341П
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296В
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493Р
flatten_23/PartitionedCallPartitionedCall+dropout_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357щ
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
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370Ј
"dropout_71/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_70/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454Ђ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall#^dropout_71/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2H
"dropout_71/StatefulPartitionedCall"dropout_71/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:         └
"
_user_specified_name
input_24
я
м
7__inference_batch_normalization_47_layer_call_fn_617940

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
■%
в
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617273

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: ћ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       б
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
 :                   h
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
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Ќ

Ж
(__inference_Predict_layer_call_fn_617742

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	ђ(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
Ћ

Ж
(__inference_Predict_layer_call_fn_617767

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	ђ(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_Predict_layer_call_and_return_conditional_losses_617572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
»

e
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         а C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         а *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         а T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ў
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:         а f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:         а "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
┬
b
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
ш
E
)__inference_MaxPool2_layer_call_fn_618022

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
љ
▒
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
 :                   z
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
 :                   o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Я
Ю
,__inference_layer_conv2_layer_call_fn_617912

inputs
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         └ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
╬&
У
C__inference_Predict_layer_call_and_return_conditional_losses_617653
input_24(
layer_conv2_617623:  
layer_conv2_617625: +
batch_normalization_47_617628: +
batch_normalization_47_617630: +
batch_normalization_47_617632: +
batch_normalization_47_617634: 

fc1_617641:	ђ(@

fc1_617643:@

fc2_617647:@

fc2_617649:
identityѕб.batch_normalization_47/StatefulPartitionedCallбfc1/StatefulPartitionedCallбfc2/StatefulPartitionedCallб#layer_conv2/StatefulPartitionedCallЃ
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_24layer_conv2_617623layer_conv2_617625*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321Ћ
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617628batch_normalization_47_617630batch_normalization_47_617632batch_normalization_47_617634*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226Э
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341П
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296▄
dropout_70/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349┌
flatten_23/PartitionedCallPartitionedCall#dropout_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357щ
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
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370┌
dropout_71/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381щ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┘
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:         └
"
_user_specified_name
input_24
гr
ч
"__inference__traced_restore_618324
file_prefix9
#assignvariableop_layer_conv2_kernel: 1
#assignvariableop_1_layer_conv2_bias: =
/assignvariableop_2_batch_normalization_47_gamma: <
.assignvariableop_3_batch_normalization_47_beta: C
5assignvariableop_4_batch_normalization_47_moving_mean: G
9assignvariableop_5_batch_normalization_47_moving_variance: 0
assignvariableop_6_fc1_kernel:	ђ(@)
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
+assignvariableop_22_sgd_fc1_kernel_momentum:	ђ(@7
)assignvariableop_23_sgd_fc1_bias_momentum:@=
+assignvariableop_24_sgd_fc2_kernel_momentum:@7
)assignvariableop_25_sgd_fc2_bias_momentum:
identity_27ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Б
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╔
value┐B╝B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHд
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOpAssignVariableOp#assignvariableop_layer_conv2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_1AssignVariableOp#assignvariableop_1_layer_conv2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_47_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_47_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_47_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_47_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_8AssignVariableOpassignvariableop_8_fc2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_9AssignVariableOpassignvariableop_9_fc2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_12AssignVariableOpassignvariableop_12_momentumIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:х
AssignVariableOp_13AssignVariableOpassignvariableop_13_sgd_iterIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_18AssignVariableOp3assignvariableop_18_sgd_layer_conv2_kernel_momentumIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_19AssignVariableOp1assignvariableop_19_sgd_layer_conv2_bias_momentumIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_20AssignVariableOp=assignvariableop_20_sgd_batch_normalization_47_gamma_momentumIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_21AssignVariableOp<assignvariableop_21_sgd_batch_normalization_47_beta_momentumIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_22AssignVariableOp+assignvariableop_22_sgd_fc1_kernel_momentumIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_23AssignVariableOp)assignvariableop_23_sgd_fc1_bias_momentumIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_24AssignVariableOp+assignvariableop_24_sgd_fc2_kernel_momentumIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_sgd_fc2_bias_momentumIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 І
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: Э
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
╚&
Т
C__inference_Predict_layer_call_and_return_conditional_losses_617401

inputs(
layer_conv2_617322:  
layer_conv2_617324: +
batch_normalization_47_617327: +
batch_normalization_47_617329: +
batch_normalization_47_617331: +
batch_normalization_47_617333: 

fc1_617371:	ђ(@

fc1_617373:@

fc2_617395:@

fc2_617397:
identityѕб.batch_normalization_47/StatefulPartitionedCallбfc1/StatefulPartitionedCallбfc2/StatefulPartitionedCallб#layer_conv2/StatefulPartitionedCallЂ
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_617322layer_conv2_617324*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617321Ћ
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_47_617327batch_normalization_47_617329batch_normalization_47_617331batch_normalization_47_617333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617226Э
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341П
MaxPool2/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_MaxPool2_layer_call_and_return_conditional_losses_617296▄
dropout_70/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349┌
flatten_23/PartitionedCallPartitionedCall#dropout_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_23_layer_call_and_return_conditional_losses_617357щ
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
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_617370┌
dropout_71/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_71_layer_call_and_return_conditional_losses_617381щ
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
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394s
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┘
NoOpNoOp/^batch_normalization_47/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         └: : : : : : : : : : 2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
ї

e
F__inference_dropout_71_layer_call_and_return_conditional_losses_617454

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
И
Љ
$__inference_fc2_layer_call_fn_618124

inputs
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_fc2_layer_call_and_return_conditional_losses_617394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Є
d
+__inference_dropout_70_layer_call_fn_618040

inputs
identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         а `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
х
G
+__inference_dropout_70_layer_call_fn_618035

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         а * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_70_layer_call_and_return_conditional_losses_617349e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         а "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs
╗
J
.__inference_activation_47_layer_call_fn_618012

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_activation_47_layer_call_and_return_conditional_losses_617341e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         └ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └ :T P
,
_output_shapes
:         └ 
 
_user_specified_nameinputs
щ
ќ
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpб"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        ѓ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         └њ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Г
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └ *
paddingSAME*
strides
Ђ
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:         └ *
squeeze_dims

§        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         └ d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         └ ё
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
»

e
F__inference_dropout_70_layer_call_and_return_conditional_losses_617493

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ќќќ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         а C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         а *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎ>Ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         а T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ў
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:         а f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:         а "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а :T P
,
_output_shapes
:         а 
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_defaultЎ
B
input_246
serving_default_input_24:0         └7
fc20
StatefulPartitionedCall:0         tensorflow/serving/predict:╣ы
═
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
П
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
Ж
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
Ц
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:_random_generator"
_tf_keras_layer
Ц
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
╝
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator"
_tf_keras_layer
╗
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
╩
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
Н
]trace_0
^trace_1
_trace_2
`trace_32Ж
(__inference_Predict_layer_call_fn_617424
(__inference_Predict_layer_call_fn_617742
(__inference_Predict_layer_call_fn_617767
(__inference_Predict_layer_call_fn_617620┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z]trace_0z^trace_1z_trace_2z`trace_3
┴
atrace_0
btrace_1
ctrace_2
dtrace_32о
C__inference_Predict_layer_call_and_return_conditional_losses_617821
C__inference_Predict_layer_call_and_return_conditional_losses_617903
C__inference_Predict_layer_call_and_return_conditional_losses_617653
C__inference_Predict_layer_call_and_return_conditional_losses_617686┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zatrace_0zbtrace_1zctrace_2zdtrace_3
═B╩
!__inference__wrapped_model_617202input_24"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л
	edecay
flearning_rate
gmomentum
hitermomentum║momentum╗$momentum╝%momentumйGmomentumЙHmomentum┐Vmomentum└Wmomentum┴"
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
Г
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
­
otrace_02М
,__inference_layer_conv2_layer_call_fn_617912б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zotrace_0
І
ptrace_02Ь
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zptrace_0
(:& 2layer_conv2/kernel
: 2layer_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
▀
vtrace_0
wtrace_12е
7__inference_batch_normalization_47_layer_call_fn_617940
7__inference_batch_normalization_47_layer_call_fn_617953│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zvtrace_0zwtrace_1
Ћ
xtrace_0
ytrace_12я
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
Ы
trace_02Н
.__inference_activation_47_layer_call_fn_618012б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
Ј
ђtrace_02­
I__inference_activation_47_layer_call_and_return_conditional_losses_618017б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zђtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ђnon_trainable_variables
ѓlayers
Ѓmetrics
 ёlayer_regularization_losses
Ёlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
№
єtrace_02л
)__inference_MaxPool2_layer_call_fn_618022б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zєtrace_0
і
Єtrace_02в
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЄtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
╦
Їtrace_0
јtrace_12љ
+__inference_dropout_70_layer_call_fn_618035
+__inference_dropout_70_layer_call_fn_618040│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0zјtrace_1
Ђ
Јtrace_0
љtrace_12к
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0zљtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ы
ќtrace_02м
+__inference_flatten_23_layer_call_fn_618062б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zќtrace_0
ї
Ќtrace_02ь
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0
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
▓
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ж
Юtrace_02╦
$__inference_fc1_layer_call_fn_618077б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЮtrace_0
Ё
ъtrace_02Т
?__inference_fc1_layer_call_and_return_conditional_losses_618088б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zъtrace_0
:	ђ(@2
fc1/kernel
:@2fc1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
╦
цtrace_0
Цtrace_12љ
+__inference_dropout_71_layer_call_fn_618093
+__inference_dropout_71_layer_call_fn_618098│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0zЦtrace_1
Ђ
дtrace_0
Дtrace_12к
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zдtrace_0zДtrace_1
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
▓
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Ж
Гtrace_02╦
$__inference_fc2_layer_call_fn_618124б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГtrace_0
Ё
«trace_02Т
?__inference_fc2_layer_call_and_return_conditional_losses_618135б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0
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
»0
░1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBЭ
(__inference_Predict_layer_call_fn_617424input_24"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_Predict_layer_call_fn_617742inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_Predict_layer_call_fn_617767inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
(__inference_Predict_layer_call_fn_617620input_24"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_Predict_layer_call_and_return_conditional_losses_617821inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_Predict_layer_call_and_return_conditional_losses_617903inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
C__inference_Predict_layer_call_and_return_conditional_losses_617653input_24"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
C__inference_Predict_layer_call_and_return_conditional_losses_617686input_24"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
╠B╔
$__inference_signature_wrapper_617717input_24"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЯBП
,__inference_layer_conv2_layer_call_fn_617912inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЧBщ
7__inference_batch_normalization_47_layer_call_fn_617940inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
7__inference_batch_normalization_47_layer_call_fn_617953inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЌBћ
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЌBћ
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
РB▀
.__inference_activation_47_layer_call_fn_618012inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
I__inference_activation_47_layer_call_and_return_conditional_losses_618017inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_MaxPool2_layer_call_fn_618022inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
­Bь
+__inference_dropout_70_layer_call_fn_618035inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
+__inference_dropout_70_layer_call_fn_618040inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▀B▄
+__inference_flatten_23_layer_call_fn_618062inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
пBН
$__inference_fc1_layer_call_fn_618077inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
зB­
?__inference_fc1_layer_call_and_return_conditional_losses_618088inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
­Bь
+__inference_dropout_71_layer_call_fn_618093inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
+__inference_dropout_71_layer_call_fn_618098inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
пBН
$__inference_fc2_layer_call_fn_618124inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
зB­
?__inference_fc2_layer_call_and_return_conditional_losses_618135inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
▒	variables
▓	keras_api

│total

┤count"
_tf_keras_metric
c
х	variables
Х	keras_api

иtotal

Иcount
╣
_fn_kwargs"
_tf_keras_metric
0
│0
┤1"
trackable_list_wrapper
.
▒	variables"
_generic_user_object
:  (2total
:  (2count
0
и0
И1"
trackable_list_wrapper
.
х	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
3:1 2SGD/layer_conv2/kernel/momentum
):' 2SGD/layer_conv2/bias/momentum
5:3 2)SGD/batch_normalization_47/gamma/momentum
4:2 2(SGD/batch_normalization_47/beta/momentum
(:&	ђ(@2SGD/fc1/kernel/momentum
!:@2SGD/fc1/bias/momentum
':%@2SGD/fc2/kernel/momentum
!:2SGD/fc2/bias/momentumн
D__inference_MaxPool2_layer_call_and_return_conditional_losses_618030ІEбB
;б8
6і3
inputs'                           
ф "Bб?
8і5
tensor_0'                           
џ «
)__inference_MaxPool2_layer_call_fn_618022ђEбB
;б8
6і3
inputs'                           
ф "7і4
unknown'                           ┴
C__inference_Predict_layer_call_and_return_conditional_losses_617653z
'$&%GHVW>б;
4б1
'і$
input_24         └
p 

 
ф ",б)
"і
tensor_0         
џ ┴
C__inference_Predict_layer_call_and_return_conditional_losses_617686z
&'$%GHVW>б;
4б1
'і$
input_24         └
p

 
ф ",б)
"і
tensor_0         
џ ┐
C__inference_Predict_layer_call_and_return_conditional_losses_617821x
'$&%GHVW<б9
2б/
%і"
inputs         └
p 

 
ф ",б)
"і
tensor_0         
џ ┐
C__inference_Predict_layer_call_and_return_conditional_losses_617903x
&'$%GHVW<б9
2б/
%і"
inputs         └
p

 
ф ",б)
"і
tensor_0         
џ Џ
(__inference_Predict_layer_call_fn_617424o
'$&%GHVW>б;
4б1
'і$
input_24         └
p 

 
ф "!і
unknown         Џ
(__inference_Predict_layer_call_fn_617620o
&'$%GHVW>б;
4б1
'і$
input_24         └
p

 
ф "!і
unknown         Ў
(__inference_Predict_layer_call_fn_617742m
'$&%GHVW<б9
2б/
%і"
inputs         └
p 

 
ф "!і
unknown         Ў
(__inference_Predict_layer_call_fn_617767m
&'$%GHVW<б9
2б/
%і"
inputs         └
p

 
ф "!і
unknown         ћ
!__inference__wrapped_model_617202o
'$&%GHVW6б3
,б)
'і$
input_24         └
ф ")ф&
$
fc2і
fc2         Х
I__inference_activation_47_layer_call_and_return_conditional_losses_618017i4б1
*б'
%і"
inputs         └ 
ф "1б.
'і$
tensor_0         └ 
џ љ
.__inference_activation_47_layer_call_fn_618012^4б1
*б'
%і"
inputs         └ 
ф "&і#
unknown         └ ┌
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_617973Ѓ'$&%@б=
6б3
-і*
inputs                   
p 
ф "9б6
/і,
tensor_0                   
џ ┌
R__inference_batch_normalization_47_layer_call_and_return_conditional_losses_618007Ѓ&'$%@б=
6б3
-і*
inputs                   
p
ф "9б6
/і,
tensor_0                   
џ │
7__inference_batch_normalization_47_layer_call_fn_617940x'$&%@б=
6б3
-і*
inputs                   
p 
ф ".і+
unknown                   │
7__inference_batch_normalization_47_layer_call_fn_617953x&'$%@б=
6б3
-і*
inputs                   
p
ф ".і+
unknown                   и
F__inference_dropout_70_layer_call_and_return_conditional_losses_618045m8б5
.б+
%і"
inputs         а 
p 
ф "1б.
'і$
tensor_0         а 
џ и
F__inference_dropout_70_layer_call_and_return_conditional_losses_618057m8б5
.б+
%і"
inputs         а 
p
ф "1б.
'і$
tensor_0         а 
џ Љ
+__inference_dropout_70_layer_call_fn_618035b8б5
.б+
%і"
inputs         а 
p 
ф "&і#
unknown         а Љ
+__inference_dropout_70_layer_call_fn_618040b8б5
.б+
%і"
inputs         а 
p
ф "&і#
unknown         а Г
F__inference_dropout_71_layer_call_and_return_conditional_losses_618103c3б0
)б&
 і
inputs         @
p 
ф ",б)
"і
tensor_0         @
џ Г
F__inference_dropout_71_layer_call_and_return_conditional_losses_618115c3б0
)б&
 і
inputs         @
p
ф ",б)
"і
tensor_0         @
џ Є
+__inference_dropout_71_layer_call_fn_618093X3б0
)б&
 і
inputs         @
p 
ф "!і
unknown         @Є
+__inference_dropout_71_layer_call_fn_618098X3б0
)б&
 і
inputs         @
p
ф "!і
unknown         @Д
?__inference_fc1_layer_call_and_return_conditional_losses_618088dGH0б-
&б#
!і
inputs         ђ(
ф ",б)
"і
tensor_0         @
џ Ђ
$__inference_fc1_layer_call_fn_618077YGH0б-
&б#
!і
inputs         ђ(
ф "!і
unknown         @д
?__inference_fc2_layer_call_and_return_conditional_losses_618135cVW/б,
%б"
 і
inputs         @
ф ",б)
"і
tensor_0         
џ ђ
$__inference_fc2_layer_call_fn_618124XVW/б,
%б"
 і
inputs         @
ф "!і
unknown         »
F__inference_flatten_23_layer_call_and_return_conditional_losses_618068e4б1
*б'
%і"
inputs         а 
ф "-б*
#і 
tensor_0         ђ(
џ Ѕ
+__inference_flatten_23_layer_call_fn_618062Z4б1
*б'
%і"
inputs         а 
ф ""і
unknown         ђ(И
G__inference_layer_conv2_layer_call_and_return_conditional_losses_617927m4б1
*б'
%і"
inputs         └
ф "1б.
'і$
tensor_0         └ 
џ њ
,__inference_layer_conv2_layer_call_fn_617912b4б1
*б'
%і"
inputs         └
ф "&і#
unknown         └ Б
$__inference_signature_wrapper_617717{
'$&%GHVWBб?
б 
8ф5
3
input_24'і$
input_24         └")ф&
$
fc2і
fc2         