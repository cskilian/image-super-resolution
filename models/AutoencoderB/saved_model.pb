σ«
Ν£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
Ύ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Ωπ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2/kernel
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
|
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*'
_output_shapes
:@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
: *
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
z
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/m

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_7/kernel/m

*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:@ *
dtype0

Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/m

*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
z
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/v

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_7/kernel/v

*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:@ *
dtype0

Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/v

*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ϊs
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*s
valuesBs Bs
ς
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
regularization_losses
	variables
	keras_api
%
#_self_saveable_object_factories
w
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
w
# _self_saveable_object_factories
!trainable_variables
"regularization_losses
#	variables
$	keras_api


%kernel
&bias
#'_self_saveable_object_factories
(trainable_variables
)regularization_losses
*	variables
+	keras_api


,kernel
-bias
#._self_saveable_object_factories
/trainable_variables
0regularization_losses
1	variables
2	keras_api
w
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api


8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<regularization_losses
=	variables
>	keras_api
w
#?_self_saveable_object_factories
@trainable_variables
Aregularization_losses
B	variables
C	keras_api


Dkernel
Ebias
#F_self_saveable_object_factories
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
w
#K_self_saveable_object_factories
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api


Pkernel
Qbias
#R_self_saveable_object_factories
Strainable_variables
Tregularization_losses
U	variables
V	keras_api


Wkernel
Xbias
#Y_self_saveable_object_factories
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
w
#^_self_saveable_object_factories
_trainable_variables
`regularization_losses
a	variables
b	keras_api
w
#c_self_saveable_object_factories
dtrainable_variables
eregularization_losses
f	variables
g	keras_api


hkernel
ibias
#j_self_saveable_object_factories
ktrainable_variables
lregularization_losses
m	variables
n	keras_api


okernel
pbias
#q_self_saveable_object_factories
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
w
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api


{kernel
|bias
#}_self_saveable_object_factories
~trainable_variables
regularization_losses
	variables
	keras_api
 
­
	iter
beta_1
beta_2

decay
learning_rate%mμ&mν,mξ-mο8mπ9mρDmςEmσPmτQmυWmφXmχhmψimωomϊpmϋ{mό|mύ%vώ&v?,v-v8v9vDvEvPvQvWvXvhvivovpv{v|v
 

%0
&1
,2
-3
84
95
D6
E7
P8
Q9
W10
X11
h12
i13
o14
p15
{16
|17
 

%0
&1
,2
-3
84
95
D6
E7
P8
Q9
W10
X11
h12
i13
o14
p15
{16
|17
²
trainable_variables
 layer_regularization_losses
regularization_losses
metrics
	variables
layer_metrics
layers
non_trainable_variables
 
 
 
 
 
²
 layer_regularization_losses
trainable_variables
regularization_losses
metrics
	variables
layer_metrics
layers
non_trainable_variables
 
 
 
 
²
 layer_regularization_losses
!trainable_variables
"regularization_losses
metrics
#	variables
layer_metrics
layers
non_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1
 

%0
&1
²
 layer_regularization_losses
(trainable_variables
)regularization_losses
metrics
*	variables
layer_metrics
layers
non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1
 

,0
-1
²
 layer_regularization_losses
/trainable_variables
0regularization_losses
metrics
1	variables
layer_metrics
layers
non_trainable_variables
 
 
 
 
²
  layer_regularization_losses
4trainable_variables
5regularization_losses
‘metrics
6	variables
’layer_metrics
£layers
€non_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
 

80
91
²
 ₯layer_regularization_losses
;trainable_variables
<regularization_losses
¦metrics
=	variables
§layer_metrics
¨layers
©non_trainable_variables
 
 
 
 
²
 ͺlayer_regularization_losses
@trainable_variables
Aregularization_losses
«metrics
B	variables
¬layer_metrics
­layers
?non_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1
 

D0
E1
²
 ―layer_regularization_losses
Gtrainable_variables
Hregularization_losses
°metrics
I	variables
±layer_metrics
²layers
³non_trainable_variables
 
 
 
 
²
 ΄layer_regularization_losses
Ltrainable_variables
Mregularization_losses
΅metrics
N	variables
Άlayer_metrics
·layers
Έnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1
 

P0
Q1
²
 Ήlayer_regularization_losses
Strainable_variables
Tregularization_losses
Ίmetrics
U	variables
»layer_metrics
Όlayers
½non_trainable_variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1
 

W0
X1
²
 Ύlayer_regularization_losses
Ztrainable_variables
[regularization_losses
Ώmetrics
\	variables
ΐlayer_metrics
Αlayers
Βnon_trainable_variables
 
 
 
 
²
 Γlayer_regularization_losses
_trainable_variables
`regularization_losses
Δmetrics
a	variables
Εlayer_metrics
Ζlayers
Ηnon_trainable_variables
 
 
 
 
²
 Θlayer_regularization_losses
dtrainable_variables
eregularization_losses
Ιmetrics
f	variables
Κlayer_metrics
Λlayers
Μnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

h0
i1
 

h0
i1
²
 Νlayer_regularization_losses
ktrainable_variables
lregularization_losses
Ξmetrics
m	variables
Οlayer_metrics
Πlayers
Ρnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1
 

o0
p1
²
 ?layer_regularization_losses
rtrainable_variables
sregularization_losses
Σmetrics
t	variables
Τlayer_metrics
Υlayers
Φnon_trainable_variables
 
 
 
 
²
 Χlayer_regularization_losses
wtrainable_variables
xregularization_losses
Ψmetrics
y	variables
Ωlayer_metrics
Ϊlayers
Ϋnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1
 

{0
|1
³
 άlayer_regularization_losses
~trainable_variables
regularization_losses
έmetrics
	variables
ήlayer_metrics
ίlayers
ΰnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

α0
β1
 

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
9
10
11
12
13
14
15
16
17
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

γtotal

δcount
ε	variables
ζ	keras_api
I

ηtotal

θcount
ι
_fn_kwargs
κ	variables
λ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

γ0
δ1

ε	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

η0
θ1

κ	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*1
_output_shapes
:?????????ΰΰ*
dtype0*&
shape:?????????ΰΰ
ώ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????ΰΰ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_10171
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ύ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_10874

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_11073ΎΧ
ύ	
ͺ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_9710

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ς	
¨
@__inference_conv2d_layer_call_and_return_conditional_losses_9599

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ύ	
ͺ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_9737

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Θ
}
(__inference_conv2d_3_layer_call_fn_10538

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_96822
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ψ	
«
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10601

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_9563

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ή
i
?__inference_add_1_layer_call_and_return_conditional_losses_9829

inputs
inputs_1
identityq
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ζ
}
(__inference_conv2d_6_layer_call_fn_10610

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_97802
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
υ	
«
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10653

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
K

G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10081

inputs
conv2d_10029
conv2d_10031
conv2d_1_10034
conv2d_1_10036
conv2d_2_10040
conv2d_2_10042
conv2d_3_10046
conv2d_3_10048
conv2d_4_10052
conv2d_4_10054
conv2d_5_10057
conv2d_5_10059
conv2d_6_10064
conv2d_6_10066
conv2d_7_10069
conv2d_7_10071
conv2d_8_10075
conv2d_8_10077
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall’ conv2d_7/StatefulPartitionedCall’ conv2d_8/StatefulPartitionedCallγ
resizing_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_resizing_2_layer_call_and_return_conditional_losses_95792
resizing_2/PartitionedCall‘
up_sampling2d_2/PartitionedCallPartitionedCall#resizing_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_95012!
up_sampling2d_2/PartitionedCallΕ
conv2d/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_10029conv2d_10031*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_95992 
conv2d/StatefulPartitionedCallΞ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_10034conv2d_1_10036*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_96262"
 conv2d_1/StatefulPartitionedCall‘
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_95132
max_pooling2d/PartitionedCallΞ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_10040conv2d_2_10042*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_96542"
 conv2d_2/StatefulPartitionedCall¨
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_95252!
max_pooling2d_1/PartitionedCallΠ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_10046conv2d_3_10048*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_96822"
 conv2d_3/StatefulPartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_95442!
up_sampling2d_3/PartitionedCallΠ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_4_10052conv2d_4_10054*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97102"
 conv2d_4/StatefulPartitionedCallΡ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_10057conv2d_5_10059*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_97372"
 conv2d_5/StatefulPartitionedCall°
add/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_97592
add/PartitionedCall
up_sampling2d_4/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_95632!
up_sampling2d_4/PartitionedCallΟ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_6_10064conv2d_6_10066*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_97802"
 conv2d_6/StatefulPartitionedCallΠ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_10069conv2d_7_10071*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_98072"
 conv2d_7/StatefulPartitionedCall³
add_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_98292
add_1/PartitionedCallΕ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_8_10075conv2d_8_10077*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_98492"
 conv2d_8/StatefulPartitionedCallΠ
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
Ώ
ω
,__inference_AutoencoderB_layer_call_fn_10120
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_100812
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1
Ί
F
*__inference_resizing_2_layer_call_fn_10458

inputs
identityΝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_resizing_2_layer_call_and_return_conditional_losses_95792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ΰΰ:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
ϋ	
«
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10509

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ϊ	
ͺ
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9654

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Β
l
@__inference_add_1_layer_call_and_return_conditional_losses_10636
inputs_0
inputs_1
identitys
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2
addu
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
χ	
ͺ
B__inference_conv2d_6_layer_call_and_return_conditional_losses_9780

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Δ
}
(__inference_conv2d_7_layer_call_fn_10630

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_98072
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Δ
}
(__inference_conv2d_1_layer_call_fn_10498

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_96262
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

Α
__inference__traced_save_10874
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_79633031562a421ebb9f95baa0a91fe6/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameθ#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*ϊ"
valueπ"Bν"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*
valueB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesΣ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Χ
_input_shapesΕ
Β: : : : @:@:@::::::::@:@:@ : : :: : : : : : : : : : : : @:@:@::::::::@:@:@ : : :: : : @:@:@::::::::@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:- )
'
_output_shapes
:@:!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::.$*
(
_output_shapes
::!%

_output_shapes	
::.&*
(
_output_shapes
::!'

_output_shapes	
::-()
'
_output_shapes
:@: )

_output_shapes
:@:,*(
&
_output_shapes
:@ : +

_output_shapes
: :,,(
&
_output_shapes
: : -

_output_shapes
::,.(
&
_output_shapes
: : /

_output_shapes
: :,0(
&
_output_shapes
: @: 1

_output_shapes
:@:-2)
'
_output_shapes
:@:!3

_output_shapes	
::.4*
(
_output_shapes
::!5

_output_shapes	
::.6*
(
_output_shapes
::!7

_output_shapes	
::.8*
(
_output_shapes
::!9

_output_shapes	
::-:)
'
_output_shapes
:@: ;

_output_shapes
:@:,<(
&
_output_shapes
:@ : =

_output_shapes
: :,>(
&
_output_shapes
: : ?

_output_shapes
::@

_output_shapes
: 
ύ
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9513

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
½
g
=__inference_add_layer_call_and_return_conditional_losses_9759

inputs
inputs_1
identityr
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,???????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,???????????????????????????:,???????????????????????????:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
¬
J
.__inference_up_sampling2d_2_layer_call_fn_9507

inputs
identityν
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_95012
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
σ	
©
A__inference_conv2d_layer_call_and_return_conditional_losses_10469

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ύ	
ͺ
B__inference_conv2d_3_layer_call_and_return_conditional_losses_9682

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ξ

π
#__inference_signature_wrapper_10171
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCall΄
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????ΰΰ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_94882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:?????????ΰΰ2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1
Ρ
`
D__inference_resizing_2_layer_call_and_return_conditional_losses_9579

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/sizeΗ
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2
resize/ResizeNearestNeighbor
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ΰΰ:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
Ύ
ω
,__inference_AutoencoderB_layer_call_fn_10022
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_99832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1

e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_9544

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
σq

G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10268

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource
identity
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_2/resize/sizeθ
'resizing_2/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_2/resize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2)
'resizing_2/resize/ResizeNearestNeighbor
up_sampling2d_2/ShapeShape8resizing_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor8resizing_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:?????????ΰΰ*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborͺ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpρ
conv2d/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
conv2d/Conv2D‘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d/Relu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΣ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_1/ReluΓ
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool±
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΧ
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_2/ReluΘ
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:?????????88*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool²
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΩ
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????882
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????882
conv2d_3/Reluy
up_sampling2d_3/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2?
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:?????????pp*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpφ
conv2d_4/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_4/Relu²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpΤ
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_5/Relu
add/addAddV2conv2d_2/Relu:activations:0conv2d_5/Relu:activations:0*
T0*0
_output_shapes
:?????????pp2	
add/addi
up_sampling2d_4/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulς
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighboradd/add:z:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:?????????ΰΰ*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpχ
conv2d_6/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
conv2d_6/Conv2D§
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_6/BiasAdd}
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_6/Relu°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_7/Conv2D/ReadVariableOpΥ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d_7/BiasAdd}
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d_7/Relu
	add_1/addAddV2conv2d/Relu:activations:0conv2d_7/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ 2
	add_1/add°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOpΗ
conv2d_8/Conv2DConv2Dadd_1/add:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ*
paddingSAME*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ2
conv2d_8/BiasAdd}
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ2
conv2d_8/Reluy
IdentityIdentityconv2d_8/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ:::::::::::::::::::Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
υ	
«
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10621

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
¨
H
,__inference_max_pooling2d_layer_call_fn_9519

inputs
identityλ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_95132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ί
Ψ
__inference__wrapped_model_9488
input_16
2autoencoderb_conv2d_conv2d_readvariableop_resource7
3autoencoderb_conv2d_biasadd_readvariableop_resource8
4autoencoderb_conv2d_1_conv2d_readvariableop_resource9
5autoencoderb_conv2d_1_biasadd_readvariableop_resource8
4autoencoderb_conv2d_2_conv2d_readvariableop_resource9
5autoencoderb_conv2d_2_biasadd_readvariableop_resource8
4autoencoderb_conv2d_3_conv2d_readvariableop_resource9
5autoencoderb_conv2d_3_biasadd_readvariableop_resource8
4autoencoderb_conv2d_4_conv2d_readvariableop_resource9
5autoencoderb_conv2d_4_biasadd_readvariableop_resource8
4autoencoderb_conv2d_5_conv2d_readvariableop_resource9
5autoencoderb_conv2d_5_biasadd_readvariableop_resource8
4autoencoderb_conv2d_6_conv2d_readvariableop_resource9
5autoencoderb_conv2d_6_biasadd_readvariableop_resource8
4autoencoderb_conv2d_7_conv2d_readvariableop_resource9
5autoencoderb_conv2d_7_biasadd_readvariableop_resource8
4autoencoderb_conv2d_8_conv2d_readvariableop_resource9
5autoencoderb_conv2d_8_biasadd_readvariableop_resource
identity
#AutoencoderB/resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2%
#AutoencoderB/resizing_2/resize/size
4AutoencoderB/resizing_2/resize/ResizeNearestNeighborResizeNearestNeighborinput_1,AutoencoderB/resizing_2/resize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(26
4AutoencoderB/resizing_2/resize/ResizeNearestNeighbor½
"AutoencoderB/up_sampling2d_2/ShapeShapeEAutoencoderB/resizing_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"AutoencoderB/up_sampling2d_2/Shape?
0AutoencoderB/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0AutoencoderB/up_sampling2d_2/strided_slice/stack²
2AutoencoderB/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_2/strided_slice/stack_1²
2AutoencoderB/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_2/strided_slice/stack_2ό
*AutoencoderB/up_sampling2d_2/strided_sliceStridedSlice+AutoencoderB/up_sampling2d_2/Shape:output:09AutoencoderB/up_sampling2d_2/strided_slice/stack:output:0;AutoencoderB/up_sampling2d_2/strided_slice/stack_1:output:0;AutoencoderB/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*AutoencoderB/up_sampling2d_2/strided_slice
"AutoencoderB/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"AutoencoderB/up_sampling2d_2/Const?
 AutoencoderB/up_sampling2d_2/mulMul3AutoencoderB/up_sampling2d_2/strided_slice:output:0+AutoencoderB/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 AutoencoderB/up_sampling2d_2/mul?
9AutoencoderB/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborEAutoencoderB/resizing_2/resize/ResizeNearestNeighbor:resized_images:0$AutoencoderB/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:?????????ΰΰ*
half_pixel_centers(2;
9AutoencoderB/up_sampling2d_2/resize/ResizeNearestNeighborΡ
)AutoencoderB/conv2d/Conv2D/ReadVariableOpReadVariableOp2autoencoderb_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)AutoencoderB/conv2d/Conv2D/ReadVariableOp₯
AutoencoderB/conv2d/Conv2DConv2DJAutoencoderB/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:01AutoencoderB/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
AutoencoderB/conv2d/Conv2DΘ
*AutoencoderB/conv2d/BiasAdd/ReadVariableOpReadVariableOp3autoencoderb_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*AutoencoderB/conv2d/BiasAdd/ReadVariableOpΪ
AutoencoderB/conv2d/BiasAddBiasAdd#AutoencoderB/conv2d/Conv2D:output:02AutoencoderB/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
AutoencoderB/conv2d/BiasAdd
AutoencoderB/conv2d/ReluRelu$AutoencoderB/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
AutoencoderB/conv2d/ReluΧ
+AutoencoderB/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+AutoencoderB/conv2d_1/Conv2D/ReadVariableOp
AutoencoderB/conv2d_1/Conv2DConv2D&AutoencoderB/conv2d/Relu:activations:03AutoencoderB/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
AutoencoderB/conv2d_1/Conv2DΞ
,AutoencoderB/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,AutoencoderB/conv2d_1/BiasAdd/ReadVariableOpβ
AutoencoderB/conv2d_1/BiasAddBiasAdd%AutoencoderB/conv2d_1/Conv2D:output:04AutoencoderB/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
AutoencoderB/conv2d_1/BiasAdd€
AutoencoderB/conv2d_1/ReluRelu&AutoencoderB/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
AutoencoderB/conv2d_1/Reluκ
"AutoencoderB/max_pooling2d/MaxPoolMaxPool(AutoencoderB/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2$
"AutoencoderB/max_pooling2d/MaxPoolΨ
+AutoencoderB/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+AutoencoderB/conv2d_2/Conv2D/ReadVariableOp
AutoencoderB/conv2d_2/Conv2DConv2D+AutoencoderB/max_pooling2d/MaxPool:output:03AutoencoderB/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
AutoencoderB/conv2d_2/Conv2DΟ
,AutoencoderB/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,AutoencoderB/conv2d_2/BiasAdd/ReadVariableOpα
AutoencoderB/conv2d_2/BiasAddBiasAdd%AutoencoderB/conv2d_2/Conv2D:output:04AutoencoderB/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_2/BiasAdd£
AutoencoderB/conv2d_2/ReluRelu&AutoencoderB/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_2/Reluο
$AutoencoderB/max_pooling2d_1/MaxPoolMaxPool(AutoencoderB/conv2d_2/Relu:activations:0*0
_output_shapes
:?????????88*
ksize
*
paddingVALID*
strides
2&
$AutoencoderB/max_pooling2d_1/MaxPoolΩ
+AutoencoderB/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+AutoencoderB/conv2d_3/Conv2D/ReadVariableOp
AutoencoderB/conv2d_3/Conv2DConv2D-AutoencoderB/max_pooling2d_1/MaxPool:output:03AutoencoderB/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88*
paddingSAME*
strides
2
AutoencoderB/conv2d_3/Conv2DΟ
,AutoencoderB/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,AutoencoderB/conv2d_3/BiasAdd/ReadVariableOpα
AutoencoderB/conv2d_3/BiasAddBiasAdd%AutoencoderB/conv2d_3/Conv2D:output:04AutoencoderB/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????882
AutoencoderB/conv2d_3/BiasAdd£
AutoencoderB/conv2d_3/ReluRelu&AutoencoderB/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????882
AutoencoderB/conv2d_3/Relu 
"AutoencoderB/up_sampling2d_3/ShapeShape(AutoencoderB/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2$
"AutoencoderB/up_sampling2d_3/Shape?
0AutoencoderB/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0AutoencoderB/up_sampling2d_3/strided_slice/stack²
2AutoencoderB/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_3/strided_slice/stack_1²
2AutoencoderB/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_3/strided_slice/stack_2ό
*AutoencoderB/up_sampling2d_3/strided_sliceStridedSlice+AutoencoderB/up_sampling2d_3/Shape:output:09AutoencoderB/up_sampling2d_3/strided_slice/stack:output:0;AutoencoderB/up_sampling2d_3/strided_slice/stack_1:output:0;AutoencoderB/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*AutoencoderB/up_sampling2d_3/strided_slice
"AutoencoderB/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"AutoencoderB/up_sampling2d_3/Const?
 AutoencoderB/up_sampling2d_3/mulMul3AutoencoderB/up_sampling2d_3/strided_slice:output:0+AutoencoderB/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2"
 AutoencoderB/up_sampling2d_3/mul΄
9AutoencoderB/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor(AutoencoderB/conv2d_3/Relu:activations:0$AutoencoderB/up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:?????????pp*
half_pixel_centers(2;
9AutoencoderB/up_sampling2d_3/resize/ResizeNearestNeighborΩ
+AutoencoderB/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+AutoencoderB/conv2d_4/Conv2D/ReadVariableOpͺ
AutoencoderB/conv2d_4/Conv2DConv2DJAutoencoderB/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:03AutoencoderB/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
AutoencoderB/conv2d_4/Conv2DΟ
,AutoencoderB/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,AutoencoderB/conv2d_4/BiasAdd/ReadVariableOpα
AutoencoderB/conv2d_4/BiasAddBiasAdd%AutoencoderB/conv2d_4/Conv2D:output:04AutoencoderB/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_4/BiasAdd£
AutoencoderB/conv2d_4/ReluRelu&AutoencoderB/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_4/ReluΩ
+AutoencoderB/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+AutoencoderB/conv2d_5/Conv2D/ReadVariableOp
AutoencoderB/conv2d_5/Conv2DConv2D(AutoencoderB/conv2d_4/Relu:activations:03AutoencoderB/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
AutoencoderB/conv2d_5/Conv2DΟ
,AutoencoderB/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,AutoencoderB/conv2d_5/BiasAdd/ReadVariableOpα
AutoencoderB/conv2d_5/BiasAddBiasAdd%AutoencoderB/conv2d_5/Conv2D:output:04AutoencoderB/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_5/BiasAdd£
AutoencoderB/conv2d_5/ReluRelu&AutoencoderB/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/conv2d_5/ReluΔ
AutoencoderB/add/addAddV2(AutoencoderB/conv2d_2/Relu:activations:0(AutoencoderB/conv2d_5/Relu:activations:0*
T0*0
_output_shapes
:?????????pp2
AutoencoderB/add/add
"AutoencoderB/up_sampling2d_4/ShapeShapeAutoencoderB/add/add:z:0*
T0*
_output_shapes
:2$
"AutoencoderB/up_sampling2d_4/Shape?
0AutoencoderB/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0AutoencoderB/up_sampling2d_4/strided_slice/stack²
2AutoencoderB/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_4/strided_slice/stack_1²
2AutoencoderB/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2AutoencoderB/up_sampling2d_4/strided_slice/stack_2ό
*AutoencoderB/up_sampling2d_4/strided_sliceStridedSlice+AutoencoderB/up_sampling2d_4/Shape:output:09AutoencoderB/up_sampling2d_4/strided_slice/stack:output:0;AutoencoderB/up_sampling2d_4/strided_slice/stack_1:output:0;AutoencoderB/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*AutoencoderB/up_sampling2d_4/strided_slice
"AutoencoderB/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"AutoencoderB/up_sampling2d_4/Const?
 AutoencoderB/up_sampling2d_4/mulMul3AutoencoderB/up_sampling2d_4/strided_slice:output:0+AutoencoderB/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2"
 AutoencoderB/up_sampling2d_4/mul¦
9AutoencoderB/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborAutoencoderB/add/add:z:0$AutoencoderB/up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:?????????ΰΰ*
half_pixel_centers(2;
9AutoencoderB/up_sampling2d_4/resize/ResizeNearestNeighborΨ
+AutoencoderB/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+AutoencoderB/conv2d_6/Conv2D/ReadVariableOp«
AutoencoderB/conv2d_6/Conv2DConv2DJAutoencoderB/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:03AutoencoderB/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
AutoencoderB/conv2d_6/Conv2DΞ
,AutoencoderB/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,AutoencoderB/conv2d_6/BiasAdd/ReadVariableOpβ
AutoencoderB/conv2d_6/BiasAddBiasAdd%AutoencoderB/conv2d_6/Conv2D:output:04AutoencoderB/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
AutoencoderB/conv2d_6/BiasAdd€
AutoencoderB/conv2d_6/ReluRelu&AutoencoderB/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
AutoencoderB/conv2d_6/ReluΧ
+AutoencoderB/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+AutoencoderB/conv2d_7/Conv2D/ReadVariableOp
AutoencoderB/conv2d_7/Conv2DConv2D(AutoencoderB/conv2d_6/Relu:activations:03AutoencoderB/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
AutoencoderB/conv2d_7/Conv2DΞ
,AutoencoderB/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,AutoencoderB/conv2d_7/BiasAdd/ReadVariableOpβ
AutoencoderB/conv2d_7/BiasAddBiasAdd%AutoencoderB/conv2d_7/Conv2D:output:04AutoencoderB/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
AutoencoderB/conv2d_7/BiasAdd€
AutoencoderB/conv2d_7/ReluRelu&AutoencoderB/conv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
AutoencoderB/conv2d_7/ReluΗ
AutoencoderB/add_1/addAddV2&AutoencoderB/conv2d/Relu:activations:0(AutoencoderB/conv2d_7/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ 2
AutoencoderB/add_1/addΧ
+AutoencoderB/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4autoencoderb_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+AutoencoderB/conv2d_8/Conv2D/ReadVariableOpϋ
AutoencoderB/conv2d_8/Conv2DConv2DAutoencoderB/add_1/add:z:03AutoencoderB/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ*
paddingSAME*
strides
2
AutoencoderB/conv2d_8/Conv2DΞ
,AutoencoderB/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5autoencoderb_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,AutoencoderB/conv2d_8/BiasAdd/ReadVariableOpβ
AutoencoderB/conv2d_8/BiasAddBiasAdd%AutoencoderB/conv2d_8/Conv2D:output:04AutoencoderB/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ2
AutoencoderB/conv2d_8/BiasAdd€
AutoencoderB/conv2d_8/ReluRelu&AutoencoderB/conv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ2
AutoencoderB/conv2d_8/Relu
IdentityIdentity(AutoencoderB/conv2d_8/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ:::::::::::::::::::Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1
τ	
ͺ
B__inference_conv2d_7_layer_call_and_return_conditional_losses_9807

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
σq

G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10365

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource
identity
resizing_2/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_2/resize/sizeθ
'resizing_2/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_2/resize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2)
'resizing_2/resize/ResizeNearestNeighbor
up_sampling2d_2/ShapeShape8resizing_2/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor8resizing_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:?????????ΰΰ*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborͺ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpρ
conv2d/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
conv2d/Conv2D‘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d/Relu°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΣ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_1/ReluΓ
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool±
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΧ
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_2/ReluΘ
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:?????????88*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool²
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΩ
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????882
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????882
conv2d_3/Reluy
up_sampling2d_3/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2?
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:?????????pp*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpφ
conv2d_4/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_4/Relu²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpΤ
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp2
conv2d_5/Relu
add/addAddV2conv2d_2/Relu:activations:0conv2d_5/Relu:activations:0*
T0*0
_output_shapes
:?????????pp2	
add/addi
up_sampling2d_4/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulς
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighboradd/add:z:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:?????????ΰΰ*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpχ
conv2d_6/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@*
paddingSAME*
strides
2
conv2d_6/Conv2D§
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_6/BiasAdd}
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ@2
conv2d_6/Relu°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_7/Conv2D/ReadVariableOpΥ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ *
paddingSAME*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d_7/BiasAdd}
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ 2
conv2d_7/Relu
	add_1/addAddV2conv2d/Relu:activations:0conv2d_7/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ 2
	add_1/add°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOpΗ
conv2d_8/Conv2DConv2Dadd_1/add:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ*
paddingSAME*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????ΰΰ2
conv2d_8/BiasAdd}
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:?????????ΰΰ2
conv2d_8/Reluy
IdentityIdentityconv2d_8/Relu:activations:0*
T0*1
_output_shapes
:?????????ΰΰ2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ:::::::::::::::::::Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
φJ
π
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9983

inputs
conv2d_9931
conv2d_9933
conv2d_1_9936
conv2d_1_9938
conv2d_2_9942
conv2d_2_9944
conv2d_3_9948
conv2d_3_9950
conv2d_4_9954
conv2d_4_9956
conv2d_5_9959
conv2d_5_9961
conv2d_6_9966
conv2d_6_9968
conv2d_7_9971
conv2d_7_9973
conv2d_8_9977
conv2d_8_9979
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall’ conv2d_7/StatefulPartitionedCall’ conv2d_8/StatefulPartitionedCallγ
resizing_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_resizing_2_layer_call_and_return_conditional_losses_95792
resizing_2/PartitionedCall‘
up_sampling2d_2/PartitionedCallPartitionedCall#resizing_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_95012!
up_sampling2d_2/PartitionedCallΓ
conv2d/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9931conv2d_9933*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_95992 
conv2d/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_9936conv2d_1_9938*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_96262"
 conv2d_1/StatefulPartitionedCall‘
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_95132
max_pooling2d/PartitionedCallΜ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_9942conv2d_2_9944*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_96542"
 conv2d_2/StatefulPartitionedCall¨
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_95252!
max_pooling2d_1/PartitionedCallΞ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_9948conv2d_3_9950*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_96822"
 conv2d_3/StatefulPartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_95442!
up_sampling2d_3/PartitionedCallΞ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_4_9954conv2d_4_9956*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97102"
 conv2d_4/StatefulPartitionedCallΟ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_9959conv2d_5_9961*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_97372"
 conv2d_5/StatefulPartitionedCall°
add/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_97592
add/PartitionedCall
up_sampling2d_4/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_95632!
up_sampling2d_4/PartitionedCallΝ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_6_9966conv2d_6_9968*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_97802"
 conv2d_6/StatefulPartitionedCallΞ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_9971conv2d_7_9973*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_98072"
 conv2d_7/StatefulPartitionedCall³
add_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_98292
add_1/PartitionedCallΓ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_8_9977conv2d_8_9979*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_98492"
 conv2d_8/StatefulPartitionedCallΠ
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
°
O
#__inference_add_layer_call_fn_10590
inputs_0
inputs_1
identityζ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_97592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,???????????????????????????:,???????????????????????????:l h
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/1
Ζ
}
(__inference_conv2d_2_layer_call_fn_10518

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_96542
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ζ
· 
!__inference__traced_restore_11073
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias&
"assignvariableop_8_conv2d_4_kernel$
 assignvariableop_9_conv2d_4_bias'
#assignvariableop_10_conv2d_5_kernel%
!assignvariableop_11_conv2d_5_bias'
#assignvariableop_12_conv2d_6_kernel%
!assignvariableop_13_conv2d_6_bias'
#assignvariableop_14_conv2d_7_kernel%
!assignvariableop_15_conv2d_7_bias'
#assignvariableop_16_conv2d_8_kernel%
!assignvariableop_17_conv2d_8_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1,
(assignvariableop_27_adam_conv2d_kernel_m*
&assignvariableop_28_adam_conv2d_bias_m.
*assignvariableop_29_adam_conv2d_1_kernel_m,
(assignvariableop_30_adam_conv2d_1_bias_m.
*assignvariableop_31_adam_conv2d_2_kernel_m,
(assignvariableop_32_adam_conv2d_2_bias_m.
*assignvariableop_33_adam_conv2d_3_kernel_m,
(assignvariableop_34_adam_conv2d_3_bias_m.
*assignvariableop_35_adam_conv2d_4_kernel_m,
(assignvariableop_36_adam_conv2d_4_bias_m.
*assignvariableop_37_adam_conv2d_5_kernel_m,
(assignvariableop_38_adam_conv2d_5_bias_m.
*assignvariableop_39_adam_conv2d_6_kernel_m,
(assignvariableop_40_adam_conv2d_6_bias_m.
*assignvariableop_41_adam_conv2d_7_kernel_m,
(assignvariableop_42_adam_conv2d_7_bias_m.
*assignvariableop_43_adam_conv2d_8_kernel_m,
(assignvariableop_44_adam_conv2d_8_bias_m,
(assignvariableop_45_adam_conv2d_kernel_v*
&assignvariableop_46_adam_conv2d_bias_v.
*assignvariableop_47_adam_conv2d_1_kernel_v,
(assignvariableop_48_adam_conv2d_1_bias_v.
*assignvariableop_49_adam_conv2d_2_kernel_v,
(assignvariableop_50_adam_conv2d_2_bias_v.
*assignvariableop_51_adam_conv2d_3_kernel_v,
(assignvariableop_52_adam_conv2d_3_bias_v.
*assignvariableop_53_adam_conv2d_4_kernel_v,
(assignvariableop_54_adam_conv2d_4_bias_v.
*assignvariableop_55_adam_conv2d_5_kernel_v,
(assignvariableop_56_adam_conv2d_5_bias_v.
*assignvariableop_57_adam_conv2d_6_kernel_v,
(assignvariableop_58_adam_conv2d_6_bias_v.
*assignvariableop_59_adam_conv2d_7_kernel_v,
(assignvariableop_60_adam_conv2d_7_bias_v.
*assignvariableop_61_adam_conv2d_8_kernel_v,
(assignvariableop_62_adam_conv2d_8_bias_v
identity_64’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ξ#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*ϊ"
valueπ"Bν"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*
valueB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesξ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3₯
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5₯
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7₯
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9₯
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18₯
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¦
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23‘
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24‘
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27°
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_7_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_7_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_8_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_8_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45°
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_conv2d_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_conv2d_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50°
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_3_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_3_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53²
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_4_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_4_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_5_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_5_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv2d_6_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58°
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_6_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59²
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_7_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60°
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_7_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_8_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62°
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_8_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΘ
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63»
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*
_input_shapes
ώ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
a
E__inference_resizing_2_layer_call_and_return_conditional_losses_10453

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/sizeΗ
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2
resize/ResizeNearestNeighbor
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ΰΰ:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
ωJ
ρ
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9923
input_1
conv2d_9871
conv2d_9873
conv2d_1_9876
conv2d_1_9878
conv2d_2_9882
conv2d_2_9884
conv2d_3_9888
conv2d_3_9890
conv2d_4_9894
conv2d_4_9896
conv2d_5_9899
conv2d_5_9901
conv2d_6_9906
conv2d_6_9908
conv2d_7_9911
conv2d_7_9913
conv2d_8_9917
conv2d_8_9919
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall’ conv2d_7/StatefulPartitionedCall’ conv2d_8/StatefulPartitionedCallδ
resizing_2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_resizing_2_layer_call_and_return_conditional_losses_95792
resizing_2/PartitionedCall‘
up_sampling2d_2/PartitionedCallPartitionedCall#resizing_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_95012!
up_sampling2d_2/PartitionedCallΓ
conv2d/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9871conv2d_9873*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_95992 
conv2d/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_9876conv2d_1_9878*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_96262"
 conv2d_1/StatefulPartitionedCall‘
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_95132
max_pooling2d/PartitionedCallΜ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_9882conv2d_2_9884*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_96542"
 conv2d_2/StatefulPartitionedCall¨
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_95252!
max_pooling2d_1/PartitionedCallΞ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_9888conv2d_3_9890*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_96822"
 conv2d_3/StatefulPartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_95442!
up_sampling2d_3/PartitionedCallΞ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_4_9894conv2d_4_9896*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97102"
 conv2d_4/StatefulPartitionedCallΟ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_9899conv2d_5_9901*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_97372"
 conv2d_5/StatefulPartitionedCall°
add/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_97592
add/PartitionedCall
up_sampling2d_4/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_95632!
up_sampling2d_4/PartitionedCallΝ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_6_9906conv2d_6_9908*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_97802"
 conv2d_6/StatefulPartitionedCallΞ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_9911conv2d_7_9913*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_98072"
 conv2d_7/StatefulPartitionedCall³
add_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_98292
add_1/PartitionedCallΓ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_8_9917conv2d_8_9919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_98492"
 conv2d_8/StatefulPartitionedCallΠ
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1
τ	
ͺ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9626

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
¬
J
.__inference_up_sampling2d_4_layer_call_fn_9569

inputs
identityν
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_95632
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Q
%__inference_add_1_layer_call_fn_10642
inputs_0
inputs_1
identityη
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_98292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:+??????????????????????????? :+??????????????????????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
¬
J
.__inference_up_sampling2d_3_layer_call_fn_9550

inputs
identityν
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_95442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
¬
J
.__inference_max_pooling2d_1_layer_call_fn_9531

inputs
identityν
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_95252
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_9501

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ό
ψ
,__inference_AutoencoderB_layer_call_fn_10447

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_100812
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
Θ
}
(__inference_conv2d_4_layer_call_fn_10558

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97102
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ώ	
«
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10569

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Δ
}
(__inference_conv2d_8_layer_call_fn_10662

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_98492
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ώ	
«
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10529

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ώ	
«
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10549

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ζ
j
>__inference_add_layer_call_and_return_conditional_losses_10584
inputs_0
inputs_1
identityt
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,???????????????????????????2
addv
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:,???????????????????????????:,???????????????????????????:l h
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/1
»
ψ
,__inference_AutoencoderB_layer_call_fn_10406

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_99832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????ΰΰ
 
_user_specified_nameinputs
ΐ
{
&__inference_conv2d_layer_call_fn_10478

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_95992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
υ	
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10489

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9525

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ωJ
ρ
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9866
input_1
conv2d_9610
conv2d_9612
conv2d_1_9637
conv2d_1_9639
conv2d_2_9665
conv2d_2_9667
conv2d_3_9693
conv2d_3_9695
conv2d_4_9721
conv2d_4_9723
conv2d_5_9748
conv2d_5_9750
conv2d_6_9791
conv2d_6_9793
conv2d_7_9818
conv2d_7_9820
conv2d_8_9860
conv2d_8_9862
identity’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’ conv2d_5/StatefulPartitionedCall’ conv2d_6/StatefulPartitionedCall’ conv2d_7/StatefulPartitionedCall’ conv2d_8/StatefulPartitionedCallδ
resizing_2/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_resizing_2_layer_call_and_return_conditional_losses_95792
resizing_2/PartitionedCall‘
up_sampling2d_2/PartitionedCallPartitionedCall#resizing_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_95012!
up_sampling2d_2/PartitionedCallΓ
conv2d/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9610conv2d_9612*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_95992 
conv2d/StatefulPartitionedCallΜ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_9637conv2d_1_9639*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_96262"
 conv2d_1/StatefulPartitionedCall‘
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_95132
max_pooling2d/PartitionedCallΜ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_9665conv2d_2_9667*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_96542"
 conv2d_2/StatefulPartitionedCall¨
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_95252!
max_pooling2d_1/PartitionedCallΞ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_9693conv2d_3_9695*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_96822"
 conv2d_3/StatefulPartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_95442!
up_sampling2d_3/PartitionedCallΞ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_4_9721conv2d_4_9723*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_97102"
 conv2d_4/StatefulPartitionedCallΟ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_9748conv2d_5_9750*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_97372"
 conv2d_5/StatefulPartitionedCall°
add/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_97592
add/PartitionedCall
up_sampling2d_4/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_95632!
up_sampling2d_4/PartitionedCallΝ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_6_9791conv2d_6_9793*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_97802"
 conv2d_6/StatefulPartitionedCallΞ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_9818conv2d_7_9820*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_98072"
 conv2d_7/StatefulPartitionedCall³
add_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_98292
add_1/PartitionedCallΓ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_8_9860conv2d_8_9862*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_98492"
 conv2d_8/StatefulPartitionedCallΠ
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:?????????ΰΰ::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:Z V
1
_output_shapes
:?????????ΰΰ
!
_user_specified_name	input_1
τ	
ͺ
B__inference_conv2d_8_layer_call_and_return_conditional_losses_9849

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Θ
}
(__inference_conv2d_5_layer_call_fn_10578

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_97372
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ώ
serving_default«
E
input_1:
serving_default_input_1:0?????????ΰΰF
conv2d_8:
StatefulPartitionedCall:0?????????ΰΰtensorflow/serving/predict:Ρ³
«
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
regularization_losses
	variables
	keras_api
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"Ϋ
_tf_keras_networkΎ{"class_name": "Functional", "name": "AutoencoderB", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "AutoencoderB", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Resizing", "config": {"name": "resizing_2", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}, "name": "resizing_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["resizing_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["conv2d_2", 0, 0, {}], ["conv2d_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["conv2d", 0, 0, {}], ["conv2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["add_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_8", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "AutoencoderB", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Resizing", "config": {"name": "resizing_2", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}, "name": "resizing_2", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["resizing_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["conv2d_2", 0, 0, {}], ["conv2d_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["conv2d", 0, 0, {}], ["conv2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["add_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_8", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
’
#_self_saveable_object_factories"ϊ
_tf_keras_input_layerΪ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"α
_tf_keras_layerΗ{"class_name": "Resizing", "name": "resizing_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resizing_2", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}
π
# _self_saveable_object_factories
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+&call_and_return_all_conditional_losses
__call__"Ί
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


%kernel
&bias
#'_self_saveable_object_factories
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Ξ	
_tf_keras_layer΄	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}



,kernel
-bias
#._self_saveable_object_factories
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 32]}}
’
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+&call_and_return_all_conditional_losses
__call__"μ
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+&call_and_return_all_conditional_losses
__call__"Π
_tf_keras_layerΆ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 64]}}
¦
#?_self_saveable_object_factories
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+&call_and_return_all_conditional_losses
 __call__"π
_tf_keras_layerΦ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



Dkernel
Ebias
#F_self_saveable_object_factories
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"Π
_tf_keras_layerΆ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 128]}}
π
#K_self_saveable_object_factories
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+£&call_and_return_all_conditional_losses
€__call__"Ί
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



Pkernel
Qbias
#R_self_saveable_object_factories
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+₯&call_and_return_all_conditional_losses
¦__call__"?
_tf_keras_layerΈ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 256]}}



Wkernel
Xbias
#Y_self_saveable_object_factories
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"?
_tf_keras_layerΈ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 128]}}
β
#^_self_saveable_object_factories
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+©&call_and_return_all_conditional_losses
ͺ__call__"¬
_tf_keras_layer{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 112, 112, 128]}, {"class_name": "TensorShape", "items": [null, 112, 112, 128]}]}
π
#c_self_saveable_object_factories
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Ί
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



hkernel
ibias
#j_self_saveable_object_factories
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+­&call_and_return_all_conditional_losses
?__call__"Ρ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 128]}}



okernel
pbias
#q_self_saveable_object_factories
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+―&call_and_return_all_conditional_losses
°__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 64]}}
δ
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+±&call_and_return_all_conditional_losses
²__call__"?
_tf_keras_layer{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 224, 224, 32]}, {"class_name": "TensorShape", "items": [null, 224, 224, 32]}]}



{kernel
|bias
#}_self_saveable_object_factories
~trainable_variables
regularization_losses
	variables
	keras_api
+³&call_and_return_all_conditional_losses
΄__call__"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 32]}}
 "
trackable_dict_wrapper
ΐ
	iter
beta_1
beta_2

decay
learning_rate%mμ&mν,mξ-mο8mπ9mρDmςEmσPmτQmυWmφXmχhmψimωomϊpmϋ{mό|mύ%vώ&v?,v-v8v9vDvEvPvQvWvXvhvivovpv{v|v"
	optimizer
-
΅serving_default"
signature_map
¦
%0
&1
,2
-3
84
95
D6
E7
P8
Q9
W10
X11
h12
i13
o14
p15
{16
|17"
trackable_list_wrapper
 "
trackable_list_wrapper
¦
%0
&1
,2
-3
84
95
D6
E7
P8
Q9
W10
X11
h12
i13
o14
p15
{16
|17"
trackable_list_wrapper
Σ
trainable_variables
 layer_regularization_losses
regularization_losses
metrics
	variables
layer_metrics
layers
non_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 layer_regularization_losses
trainable_variables
regularization_losses
metrics
	variables
layer_metrics
layers
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 layer_regularization_losses
!trainable_variables
"regularization_losses
metrics
#	variables
layer_metrics
layers
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_dict_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
΅
 layer_regularization_losses
(trainable_variables
)regularization_losses
metrics
*	variables
layer_metrics
layers
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
΅
 layer_regularization_losses
/trainable_variables
0regularization_losses
metrics
1	variables
layer_metrics
layers
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
  layer_regularization_losses
4trainable_variables
5regularization_losses
‘metrics
6	variables
’layer_metrics
£layers
€non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
΅
 ₯layer_regularization_losses
;trainable_variables
<regularization_losses
¦metrics
=	variables
§layer_metrics
¨layers
©non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 ͺlayer_regularization_losses
@trainable_variables
Aregularization_losses
«metrics
B	variables
¬layer_metrics
­layers
?non_trainable_variables
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_3/kernel
:2conv2d_3/bias
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
΅
 ―layer_regularization_losses
Gtrainable_variables
Hregularization_losses
°metrics
I	variables
±layer_metrics
²layers
³non_trainable_variables
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 ΄layer_regularization_losses
Ltrainable_variables
Mregularization_losses
΅metrics
N	variables
Άlayer_metrics
·layers
Έnon_trainable_variables
€__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
΅
 Ήlayer_regularization_losses
Strainable_variables
Tregularization_losses
Ίmetrics
U	variables
»layer_metrics
Όlayers
½non_trainable_variables
¦__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
΅
 Ύlayer_regularization_losses
Ztrainable_variables
[regularization_losses
Ώmetrics
\	variables
ΐlayer_metrics
Αlayers
Βnon_trainable_variables
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 Γlayer_regularization_losses
_trainable_variables
`regularization_losses
Δmetrics
a	variables
Εlayer_metrics
Ζlayers
Ηnon_trainable_variables
ͺ__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 Θlayer_regularization_losses
dtrainable_variables
eregularization_losses
Ιmetrics
f	variables
Κlayer_metrics
Λlayers
Μnon_trainable_variables
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_6/kernel
:@2conv2d_6/bias
 "
trackable_dict_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
΅
 Νlayer_regularization_losses
ktrainable_variables
lregularization_losses
Ξmetrics
m	variables
Οlayer_metrics
Πlayers
Ρnon_trainable_variables
?__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
):'@ 2conv2d_7/kernel
: 2conv2d_7/bias
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
΅
 ?layer_regularization_losses
rtrainable_variables
sregularization_losses
Σmetrics
t	variables
Τlayer_metrics
Υlayers
Φnon_trainable_variables
°__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 Χlayer_regularization_losses
wtrainable_variables
xregularization_losses
Ψmetrics
y	variables
Ωlayer_metrics
Ϊlayers
Ϋnon_trainable_variables
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_8/kernel
:2conv2d_8/bias
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
Ά
 άlayer_regularization_losses
~trainable_variables
regularization_losses
έmetrics
	variables
ήlayer_metrics
ίlayers
ΰnon_trainable_variables
΄__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
α0
β1"
trackable_list_wrapper
 "
trackable_dict_wrapper
¦
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
9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ

γtotal

δcount
ε	variables
ζ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ηtotal

θcount
ι
_fn_kwargs
κ	variables
λ	keras_api"Κ
_tf_keras_metric―{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
0
γ0
δ1"
trackable_list_wrapper
.
ε	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
η0
θ1"
trackable_list_wrapper
.
κ	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
/:-@2Adam/conv2d_2/kernel/m
!:2Adam/conv2d_2/bias/m
0:.2Adam/conv2d_3/kernel/m
!:2Adam/conv2d_3/bias/m
0:.2Adam/conv2d_4/kernel/m
!:2Adam/conv2d_4/bias/m
0:.2Adam/conv2d_5/kernel/m
!:2Adam/conv2d_5/bias/m
/:-@2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
.:,@ 2Adam/conv2d_7/kernel/m
 : 2Adam/conv2d_7/bias/m
.:, 2Adam/conv2d_8/kernel/m
 :2Adam/conv2d_8/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
/:-@2Adam/conv2d_2/kernel/v
!:2Adam/conv2d_2/bias/v
0:.2Adam/conv2d_3/kernel/v
!:2Adam/conv2d_3/bias/v
0:.2Adam/conv2d_4/kernel/v
!:2Adam/conv2d_4/bias/v
0:.2Adam/conv2d_5/kernel/v
!:2Adam/conv2d_5/bias/v
/:-@2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
.:,@ 2Adam/conv2d_7/kernel/v
 : 2Adam/conv2d_7/bias/v
.:, 2Adam/conv2d_8/kernel/v
 :2Adam/conv2d_8/bias/v
η2δ
__inference__wrapped_model_9488ΐ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *0’-
+(
input_1?????????ΰΰ
θ2ε
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9866
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10365
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10268
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9923ΐ
·²³
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

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ώ2ϋ
,__inference_AutoencoderB_layer_call_fn_10120
,__inference_AutoencoderB_layer_call_fn_10447
,__inference_AutoencoderB_layer_call_fn_10406
,__inference_AutoencoderB_layer_call_fn_10022ΐ
·²³
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

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ο2μ
E__inference_resizing_2_layer_call_and_return_conditional_losses_10453’
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
annotationsͺ *
 
Τ2Ρ
*__inference_resizing_2_layer_call_fn_10458’
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
annotationsͺ *
 
±2?
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_9501ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
.__inference_up_sampling2d_2_layer_call_fn_9507ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
λ2θ
A__inference_conv2d_layer_call_and_return_conditional_losses_10469’
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
annotationsͺ *
 
Π2Ν
&__inference_conv2d_layer_call_fn_10478’
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
annotationsͺ *
 
ν2κ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10489’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_1_layer_call_fn_10498’
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
annotationsͺ *
 
―2¬
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9513ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
,__inference_max_pooling2d_layer_call_fn_9519ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ν2κ
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10509’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_2_layer_call_fn_10518’
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
annotationsͺ *
 
±2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9525ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
.__inference_max_pooling2d_1_layer_call_fn_9531ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ν2κ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10529’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_3_layer_call_fn_10538’
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
annotationsͺ *
 
±2?
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_9544ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
.__inference_up_sampling2d_3_layer_call_fn_9550ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ν2κ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10549’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_4_layer_call_fn_10558’
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
annotationsͺ *
 
ν2κ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10569’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_5_layer_call_fn_10578’
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
annotationsͺ *
 
θ2ε
>__inference_add_layer_call_and_return_conditional_losses_10584’
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
annotationsͺ *
 
Ν2Κ
#__inference_add_layer_call_fn_10590’
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
annotationsͺ *
 
±2?
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_9563ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
.__inference_up_sampling2d_4_layer_call_fn_9569ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ν2κ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10601’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_6_layer_call_fn_10610’
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
annotationsͺ *
 
ν2κ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10621’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_7_layer_call_fn_10630’
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
annotationsͺ *
 
κ2η
@__inference_add_1_layer_call_and_return_conditional_losses_10636’
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
annotationsͺ *
 
Ο2Μ
%__inference_add_1_layer_call_fn_10642’
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
annotationsͺ *
 
ν2κ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10653’
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
annotationsͺ *
 
?2Ο
(__inference_conv2d_8_layer_call_fn_10662’
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
annotationsͺ *
 
2B0
#__inference_signature_wrapper_10171input_1Τ
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10268%&,-89DEPQWXhiop{|A’>
7’4
*'
inputs?????????ΰΰ
p

 
ͺ "/’,
%"
0?????????ΰΰ
 Τ
G__inference_AutoencoderB_layer_call_and_return_conditional_losses_10365%&,-89DEPQWXhiop{|A’>
7’4
*'
inputs?????????ΰΰ
p 

 
ͺ "/’,
%"
0?????????ΰΰ
 δ
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9866%&,-89DEPQWXhiop{|B’?
8’5
+(
input_1?????????ΰΰ
p

 
ͺ "?’<
52
0+???????????????????????????
 δ
F__inference_AutoencoderB_layer_call_and_return_conditional_losses_9923%&,-89DEPQWXhiop{|B’?
8’5
+(
input_1?????????ΰΰ
p 

 
ͺ "?’<
52
0+???????????????????????????
 ½
,__inference_AutoencoderB_layer_call_fn_10022%&,-89DEPQWXhiop{|B’?
8’5
+(
input_1?????????ΰΰ
p

 
ͺ "2/+???????????????????????????½
,__inference_AutoencoderB_layer_call_fn_10120%&,-89DEPQWXhiop{|B’?
8’5
+(
input_1?????????ΰΰ
p 

 
ͺ "2/+???????????????????????????Ό
,__inference_AutoencoderB_layer_call_fn_10406%&,-89DEPQWXhiop{|A’>
7’4
*'
inputs?????????ΰΰ
p

 
ͺ "2/+???????????????????????????Ό
,__inference_AutoencoderB_layer_call_fn_10447%&,-89DEPQWXhiop{|A’>
7’4
*'
inputs?????????ΰΰ
p 

 
ͺ "2/+???????????????????????????³
__inference__wrapped_model_9488%&,-89DEPQWXhiop{|:’7
0’-
+(
input_1?????????ΰΰ
ͺ "=ͺ:
8
conv2d_8,)
conv2d_8?????????ΰΰ
@__inference_add_1_layer_call_and_return_conditional_losses_10636Υ’
’
|
<9
inputs/0+??????????????????????????? 
<9
inputs/1+??????????????????????????? 
ͺ "?’<
52
0+??????????????????????????? 
 ς
%__inference_add_1_layer_call_fn_10642Θ’
’
|
<9
inputs/0+??????????????????????????? 
<9
inputs/1+??????????????????????????? 
ͺ "2/+??????????????????????????? 
>__inference_add_layer_call_and_return_conditional_losses_10584Ω’
’
~
=:
inputs/0,???????????????????????????
=:
inputs/1,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 τ
#__inference_add_layer_call_fn_10590Μ’
’
~
=:
inputs/0,???????????????????????????
=:
inputs/1,???????????????????????????
ͺ "30,???????????????????????????Ψ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10489,-I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+???????????????????????????@
 °
(__inference_conv2d_1_layer_call_fn_10498,-I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+???????????????????????????@Ω
C__inference_conv2d_2_layer_call_and_return_conditional_losses_1050989I’F
?’<
:7
inputs+???????????????????????????@
ͺ "@’=
63
0,???????????????????????????
 ±
(__inference_conv2d_2_layer_call_fn_1051889I’F
?’<
:7
inputs+???????????????????????????@
ͺ "30,???????????????????????????Ϊ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10529DEJ’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 ²
(__inference_conv2d_3_layer_call_fn_10538DEJ’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????Ϊ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10549PQJ’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 ²
(__inference_conv2d_4_layer_call_fn_10558PQJ’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????Ϊ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10569WXJ’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 ²
(__inference_conv2d_5_layer_call_fn_10578WXJ’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????Ω
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10601hiJ’G
@’=
;8
inputs,???????????????????????????
ͺ "?’<
52
0+???????????????????????????@
 ±
(__inference_conv2d_6_layer_call_fn_10610hiJ’G
@’=
;8
inputs,???????????????????????????
ͺ "2/+???????????????????????????@Ψ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10621opI’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+??????????????????????????? 
 °
(__inference_conv2d_7_layer_call_fn_10630opI’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+??????????????????????????? Ψ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10653{|I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+???????????????????????????
 °
(__inference_conv2d_8_layer_call_fn_10662{|I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+???????????????????????????Φ
A__inference_conv2d_layer_call_and_return_conditional_losses_10469%&I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+??????????????????????????? 
 ?
&__inference_conv2d_layer_call_fn_10478%&I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+??????????????????????????? μ
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9525R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_max_pooling2d_1_layer_call_fn_9531R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????κ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_9513R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Β
,__inference_max_pooling2d_layer_call_fn_9519R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????³
E__inference_resizing_2_layer_call_and_return_conditional_losses_10453j9’6
/’,
*'
inputs?????????ΰΰ
ͺ "-’*
# 
0?????????88
 
*__inference_resizing_2_layer_call_fn_10458]9’6
/’,
*'
inputs?????????ΰΰ
ͺ " ?????????88Β
#__inference_signature_wrapper_10171%&,-89DEPQWXhiop{|E’B
’ 
;ͺ8
6
input_1+(
input_1?????????ΰΰ"=ͺ:
8
conv2d_8,)
conv2d_8?????????ΰΰμ
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_9501R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_up_sampling2d_2_layer_call_fn_9507R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????μ
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_9544R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_up_sampling2d_3_layer_call_fn_9550R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????μ
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_9563R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_up_sampling2d_4_layer_call_fn_9569R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????