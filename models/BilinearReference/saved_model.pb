??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
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

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
pipeline
	optimizer

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?
	layer-0

layer-1
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
 
 
 
?
metrics
non_trainable_variables
	variables
layer_regularization_losses
regularization_losses
trainable_variables

layers
layer_metrics
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
 
?
metrics
 non_trainable_variables
	variables
!layer_regularization_losses
regularization_losses
trainable_variables

"layers
#layer_metrics

$0
%1
 
 

0
 
 
 
 
 
?
&metrics
'non_trainable_variables
	variables
(layer_regularization_losses
regularization_losses
trainable_variables

)layers
*layer_metrics
 
 
 
 
?
+metrics
,non_trainable_variables
	variables
-layer_regularization_losses
regularization_losses
trainable_variables

.layers
/layer_metrics
 
 
 

	0

1
 
4
	0total
	1count
2	variables
3	keras_api
D
	4total
	5count
6
_fn_kwargs
7	variables
8	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

00
11

2	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

7	variables
?
serving_default_xPlaceholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
?
PartitionedCallPartitionedCallserving_default_x*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_12330
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin

2*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_12482
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1*
Tin	
2*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_12504??
?
N
1__inference_bilinear_reference_layer_call_fn_6684
input_1
identity?
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_66792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:s o
J
_output_shapes8
6:4????????????????????????????????????
!
_user_specified_name	input_1
?

c
G__inference_sequential_1_layer_call_and_return_conditional_losses_12395

inputs
identity?
resizing_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????88?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_123592
resizing_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_123432!
up_sampling2d_1/PartitionedCall?
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0*
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
?
H
1__inference_bilinear_reference_layer_call_fn_6689
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_66792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
R
,__inference_sequential_1_layer_call_fn_12398
resizing_1_input
identity?
PartitionedCallPartitionedCallresizing_1_input*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_123952
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:| x
J
_output_shapes8
6:4????????????????????????????????????
*
_user_specified_nameresizing_1_input
?
H
,__inference_sequential_1_layer_call_fn_12431

inputs
identity?
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_123842
PartitionedCall?
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
?
:
#__inference_signature_wrapper_12330
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_123232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
c
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6679
x
identity?
sequential_1/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_66742
sequential_1/PartitionedCall?
IdentityIdentity%sequential_1/PartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?

m
G__inference_sequential_1_layer_call_and_return_conditional_losses_12375
resizing_1_input
identity?
resizing_1/PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????88?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_123592
resizing_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_123432!
up_sampling2d_1/PartitionedCall?
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:| x
J
_output_shapes8
6:4????????????????????????????????????
*
_user_specified_nameresizing_1_input
?
R
,__inference_sequential_1_layer_call_fn_12387
resizing_1_input
identity?
PartitionedCallPartitionedCallresizing_1_input*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_123842
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:| x
J
_output_shapes8
6:4????????????????????????????????????
*
_user_specified_nameresizing_1_input
?
c
G__inference_sequential_1_layer_call_and_return_conditional_losses_12426

inputs
identity?
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_1/resize/size?
'resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_1/resize/ResizeNearestNeighbor?
up_sampling2d_1/ShapeShape8resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinear8resizing_1/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentity6up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
H
1__inference_bilinear_reference_layer_call_fn_6699
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_66792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
f
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_12343

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
strided_slice/stack_2?
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
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
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
?
K
/__inference_up_sampling2d_1_layer_call_fn_12349

inputs
identity?
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
GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_123432
PartitionedCall?
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
?
b
F__inference_sequential_1_layer_call_and_return_conditional_losses_6674

inputs
identity?
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_1/resize/size?
'resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_1/resize/ResizeNearestNeighbor?
up_sampling2d_1/ShapeShape8resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinear8resizing_1/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentity6up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

c
G__inference_sequential_1_layer_call_and_return_conditional_losses_12384

inputs
identity?
resizing_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????88?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_123592
resizing_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_123432!
up_sampling2d_1/PartitionedCall?
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0*
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
?
c
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6606
x
identity?
#sequential_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2%
#sequential_1/resizing_1/resize/size?
4sequential_1/resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborx,sequential_1/resizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(26
4sequential_1/resizing_1/resize/ResizeNearestNeighbor?
"sequential_1/up_sampling2d_1/ShapeShapeEsequential_1/resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"sequential_1/up_sampling2d_1/Shape?
0sequential_1/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/up_sampling2d_1/strided_slice/stack?
2sequential_1/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/up_sampling2d_1/strided_slice/stack_1?
2sequential_1/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/up_sampling2d_1/strided_slice/stack_2?
*sequential_1/up_sampling2d_1/strided_sliceStridedSlice+sequential_1/up_sampling2d_1/Shape:output:09sequential_1/up_sampling2d_1/strided_slice/stack:output:0;sequential_1/up_sampling2d_1/strided_slice/stack_1:output:0;sequential_1/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_1/up_sampling2d_1/strided_slice?
"sequential_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_1/Const?
 sequential_1/up_sampling2d_1/mulMul3sequential_1/up_sampling2d_1/strided_slice:output:0+sequential_1/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_1/mul?
2sequential_1/up_sampling2d_1/resize/ResizeBilinearResizeBilinearEsequential_1/resizing_1/resize/ResizeNearestNeighbor:resized_images:0$sequential_1/up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(24
2sequential_1/up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentityCsequential_1/up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
N
1__inference_bilinear_reference_layer_call_fn_6694
input_1
identity?
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_66792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:s o
J
_output_shapes8
6:4????????????????????????????????????
!
_user_specified_name	input_1
?
a
E__inference_resizing_1_layer_call_and_return_conditional_losses_12359

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/size?
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*8
_output_shapes&
$:"?????????88?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6485
x
identity?
#sequential_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2%
#sequential_1/resizing_1/resize/size?
4sequential_1/resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborx,sequential_1/resizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(26
4sequential_1/resizing_1/resize/ResizeNearestNeighbor?
"sequential_1/up_sampling2d_1/ShapeShapeEsequential_1/resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"sequential_1/up_sampling2d_1/Shape?
0sequential_1/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_1/up_sampling2d_1/strided_slice/stack?
2sequential_1/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/up_sampling2d_1/strided_slice/stack_1?
2sequential_1/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/up_sampling2d_1/strided_slice/stack_2?
*sequential_1/up_sampling2d_1/strided_sliceStridedSlice+sequential_1/up_sampling2d_1/Shape:output:09sequential_1/up_sampling2d_1/strided_slice/stack:output:0;sequential_1/up_sampling2d_1/strided_slice/stack_1:output:0;sequential_1/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_1/up_sampling2d_1/strided_slice?
"sequential_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_1/Const?
 sequential_1/up_sampling2d_1/mulMul3sequential_1/up_sampling2d_1/strided_slice:output:0+sequential_1/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_1/mul?
2sequential_1/up_sampling2d_1/resize/ResizeBilinearResizeBilinearEsequential_1/resizing_1/resize/ResizeNearestNeighbor:resized_images:0$sequential_1/up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(24
2sequential_1/up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentityCsequential_1/up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
7
 __inference__wrapped_model_12323
x
identity?
"bilinear_reference/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_117632$
"bilinear_reference/PartitionedCall?
IdentityIdentity+bilinear_reference/PartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
c
G__inference_sequential_1_layer_call_and_return_conditional_losses_12412

inputs
identity?
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_1/resize/size?
'resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_1/resize/ResizeNearestNeighbor?
up_sampling2d_1/ShapeShape8resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinear8resizing_1/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentity6up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_12482
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c60d52422e6747b6981f034c839007f8/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_sequential_1_layer_call_fn_12436

inputs
identity?
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_123952
PartitionedCall?
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
?
i
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6704
input_1
identity?
sequential_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_66742
sequential_1/PartitionedCall?
IdentityIdentity%sequential_1/PartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:s o
J
_output_shapes8
6:4????????????????????????????????????
!
_user_specified_name	input_1
?
F
*__inference_resizing_1_layer_call_fn_12447

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????88?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_123592
PartitionedCall}
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"?????????88?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
i
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6654
input_1
identity?
sequential_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_66492
sequential_1/PartitionedCall?
IdentityIdentity%sequential_1/PartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:s o
J
_output_shapes8
6:4????????????????????????????????????
!
_user_specified_name	input_1
?
a
E__inference_resizing_1_layer_call_and_return_conditional_losses_12442

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/size?
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*8
_output_shapes&
$:"?????????88?????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
!__inference__traced_restore_12504
file_prefix
assignvariableop_total
assignvariableop_1_count
assignvariableop_2_total_1
assignvariableop_3_count_1

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_totalIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_countIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_restored_function_body_11763
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*:
_output_shapes(
&:$????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_66062
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:m i
J
_output_shapes8
6:4????????????????????????????????????

_user_specified_namex
?
b
F__inference_sequential_1_layer_call_and_return_conditional_losses_6649

inputs
identity?
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_1/resize/size?
'resizing_1/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_1/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_1/resize/ResizeNearestNeighbor?
up_sampling2d_1/ShapeShape8resizing_1/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinear8resizing_1/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_1/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinear?
IdentityIdentity6up_sampling2d_1/resize/ResizeBilinear:resized_images:0*
T0*:
_output_shapes(
&:$????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

m
G__inference_sequential_1_layer_call_and_return_conditional_losses_12369
resizing_1_input
identity?
resizing_1/PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????88?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_1_layer_call_and_return_conditional_losses_123592
resizing_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall#resizing_1/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_123432!
up_sampling2d_1/PartitionedCall?
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:| x
J
_output_shapes8
6:4????????????????????????????????????
*
_user_specified_nameresizing_1_input"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
R
xM
serving_default_x:04????????????????????????????????????G
output_1;
PartitionedCall:0$????????????????????tensorflow/serving/predict:?w
?
pipeline
	optimizer

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
9_default_save_signature
:__call__
*;&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "BilinearReference", "name": "bilinear_reference", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BilinearReference"}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	layer-0

layer-1
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
<__call__
*=&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "bilinear"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "bilinear"}}]}}}
"
	optimizer
,
>serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
non_trainable_variables
	variables
layer_regularization_losses
regularization_losses
trainable_variables

layers
layer_metrics
:__call__
9_default_save_signature
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Resizing", "name": "resizing_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}
?
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "bilinear"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
 non_trainable_variables
	variables
!layer_regularization_losses
regularization_losses
trainable_variables

"layers
#layer_metrics
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
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
?
&metrics
'non_trainable_variables
	variables
(layer_regularization_losses
regularization_losses
trainable_variables

)layers
*layer_metrics
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+metrics
,non_trainable_variables
	variables
-layer_regularization_losses
regularization_losses
trainable_variables

.layers
/layer_metrics
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	0total
	1count
2	variables
3	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	4total
	5count
6
_fn_kwargs
7	variables
8	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
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
:  (2total
:  (2count
.
00
11"
trackable_list_wrapper
-
2	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
-
7	variables"
_generic_user_object
?2?
 __inference__wrapped_model_12323?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *C?@
>?;
x4????????????????????????????????????
?2?
1__inference_bilinear_reference_layer_call_fn_6694
1__inference_bilinear_reference_layer_call_fn_6689
1__inference_bilinear_reference_layer_call_fn_6684
1__inference_bilinear_reference_layer_call_fn_6699?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6606
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6485
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6654
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6704?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_12398
,__inference_sequential_1_layer_call_fn_12431
,__inference_sequential_1_layer_call_fn_12387
,__inference_sequential_1_layer_call_fn_12436?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12369
G__inference_sequential_1_layer_call_and_return_conditional_losses_12426
G__inference_sequential_1_layer_call_and_return_conditional_losses_12412
G__inference_sequential_1_layer_call_and_return_conditional_losses_12375?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
,B*
#__inference_signature_wrapper_12330x
?2?
*__inference_resizing_1_layer_call_fn_12447?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_resizing_1_layer_call_and_return_conditional_losses_12442?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling2d_1_layer_call_fn_12349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_12343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84?????????????????????????????????????
 __inference__wrapped_model_12323?M?J
C?@
>?;
x4????????????????????????????????????
? "F?C
A
output_15?2
output_1$?????????????????????
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6485?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6606?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6654?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
L__inference_bilinear_reference_layer_call_and_return_conditional_losses_6704?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
1__inference_bilinear_reference_layer_call_fn_6684?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "+?($?????????????????????
1__inference_bilinear_reference_layer_call_fn_6689?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "+?($?????????????????????
1__inference_bilinear_reference_layer_call_fn_6694?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "+?($?????????????????????
1__inference_bilinear_reference_layer_call_fn_6699?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "+?($?????????????????????
E__inference_resizing_1_layer_call_and_return_conditional_losses_12442?R?O
H?E
C?@
inputs4????????????????????????????????????
? "6?3
,?)
0"?????????88?????????
? ?
*__inference_resizing_1_layer_call_fn_12447R?O
H?E
C?@
inputs4????????????????????????????????????
? ")?&"?????????88??????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_12369?d?a
Z?W
M?J
resizing_1_input4????????????????????????????????????
p

 
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12375?d?a
Z?W
M?J
resizing_1_input4????????????????????????????????????
p 

 
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12412?Z?W
P?M
C?@
inputs4????????????????????????????????????
p

 
? "8?5
.?+
0$????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12426?Z?W
P?M
C?@
inputs4????????????????????????????????????
p 

 
? "8?5
.?+
0$????????????????????
? ?
,__inference_sequential_1_layer_call_fn_12387?d?a
Z?W
M?J
resizing_1_input4????????????????????????????????????
p

 
? ";?84?????????????????????????????????????
,__inference_sequential_1_layer_call_fn_12398?d?a
Z?W
M?J
resizing_1_input4????????????????????????????????????
p 

 
? ";?84?????????????????????????????????????
,__inference_sequential_1_layer_call_fn_12431?Z?W
P?M
C?@
inputs4????????????????????????????????????
p

 
? ";?84?????????????????????????????????????
,__inference_sequential_1_layer_call_fn_12436?Z?W
P?M
C?@
inputs4????????????????????????????????????
p 

 
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_12330?R?O
? 
H?E
C
x>?;
x4????????????????????????????????????"F?C
A
output_15?2
output_1$?????????????????????
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_12343?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_1_layer_call_fn_12349?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????