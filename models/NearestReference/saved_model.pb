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
GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_6268
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
GPU2*0J 8? *&
f!R
__inference__traced_save_6420
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
GPU2*0J 8? *)
f$R"
 __inference__traced_restore_6442??
?
6
__inference__wrapped_model_6261
x
identity?
!nearest_reference/PartitionedCallPartitionedCallx*
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
GPU2*0J 8? *0
f+R)
'__inference_restored_function_body_57012#
!nearest_reference/PartitionedCall?
IdentityIdentity*nearest_reference/PartitionedCall:output:0*
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
?
a
J__inference_nearest_reference_layer_call_and_return_conditional_losses_506
x
identity?
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2!
sequential/resizing/resize/size?
0sequential/resizing/resize/ResizeNearestNeighborResizeNearestNeighborx(sequential/resizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(22
0sequential/resizing/resize/ResizeNearestNeighbor?
sequential/up_sampling2d/ShapeShapeAsequential/resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2 
sequential/up_sampling2d/Shape?
,sequential/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/up_sampling2d/strided_slice/stack?
.sequential/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/up_sampling2d/strided_slice/stack_1?
.sequential/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/up_sampling2d/strided_slice/stack_2?
&sequential/up_sampling2d/strided_sliceStridedSlice'sequential/up_sampling2d/Shape:output:05sequential/up_sampling2d/strided_slice/stack:output:07sequential/up_sampling2d/strided_slice/stack_1:output:07sequential/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2(
&sequential/up_sampling2d/strided_slice?
sequential/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
sequential/up_sampling2d/Const?
sequential/up_sampling2d/mulMul/sequential/up_sampling2d/strided_slice:output:0'sequential/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
sequential/up_sampling2d/mul?
5sequential/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborAsequential/resizing/resize/ResizeNearestNeighbor:resized_images:0 sequential/up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(27
5sequential/up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentityFsequential/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
E
)__inference_sequential_layer_call_fn_6374

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
GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63332
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
L
/__inference_nearest_reference_layer_call_fn_647
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
GPU2*0J 8? *S
fNRL
J__inference_nearest_reference_layer_call_and_return_conditional_losses_6422
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
?
`
D__inference_sequential_layer_call_and_return_conditional_losses_6350

inputs
identity}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing/resize/size?
%resizing/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2'
%resizing/resize/ResizeNearestNeighbor?
up_sampling2d/ShapeShape6resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor6resizing/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
M
)__inference_sequential_layer_call_fn_6325
resizing_input
identity?
PartitionedCallPartitionedCallresizing_input*
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
GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63222
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:z v
J
_output_shapes8
6:4????????????????????????????????????
(
_user_specified_nameresizing_input
?
a
J__inference_nearest_reference_layer_call_and_return_conditional_losses_642
x
identity?
sequential/PartitionedCallPartitionedCallx*
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
GPU2*0J 8? *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_6322
sequential/PartitionedCall?
IdentityIdentity#sequential/PartitionedCall:output:0*
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
g
J__inference_nearest_reference_layer_call_and_return_conditional_losses_637
input_1
identity?
sequential/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_6322
sequential/PartitionedCall?
IdentityIdentity#sequential/PartitionedCall:output:0*
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
/__inference_nearest_reference_layer_call_fn_652
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
GPU2*0J 8? *S
fNRL
J__inference_nearest_reference_layer_call_and_return_conditional_losses_6422
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
?
9
"__inference_signature_wrapper_6268
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
GPU2*0J 8? *(
f#R!
__inference__wrapped_model_62612
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
M
)__inference_sequential_layer_call_fn_6336
resizing_input
identity?
PartitionedCallPartitionedCallresizing_input*
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
GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63332
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:z v
J
_output_shapes8
6:4????????????????????????????????????
(
_user_specified_nameresizing_input
?
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6281

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
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
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
?
>
'__inference_restored_function_body_5701
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
GPU2*0J 8? *S
fNRL
J__inference_nearest_reference_layer_call_and_return_conditional_losses_5062
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
^
B__inference_resizing_layer_call_and_return_conditional_losses_6380

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
?
C
'__inference_resizing_layer_call_fn_6385

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
GPU2*0J 8? *K
fFRD
B__inference_resizing_layer_call_and_return_conditional_losses_62972
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
?
?
 __inference__traced_restore_6442
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
?
^
B__inference_resizing_layer_call_and_return_conditional_losses_6297

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
?	
`
D__inference_sequential_layer_call_and_return_conditional_losses_6333

inputs
identity?
resizing/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_resizing_layer_call_and_return_conditional_losses_62972
resizing/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_62812
up_sampling2d/PartitionedCall?
IdentityIdentity&up_sampling2d/PartitionedCall:output:0*
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
?
?
__inference__traced_save_6420
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
value3B1 B+_temp_fbf395d9e8ed481a81626bfa52b7a191/part2	
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
?
_
C__inference_sequential_layer_call_and_return_conditional_losses_607

inputs
identity}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing/resize/size?
%resizing/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2'
%resizing/resize/ResizeNearestNeighbor?
up_sampling2d/ShapeShape6resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor6resizing/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
?
a
J__inference_nearest_reference_layer_call_and_return_conditional_losses_443
x
identity?
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2!
sequential/resizing/resize/size?
0sequential/resizing/resize/ResizeNearestNeighborResizeNearestNeighborx(sequential/resizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(22
0sequential/resizing/resize/ResizeNearestNeighbor?
sequential/up_sampling2d/ShapeShapeAsequential/resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2 
sequential/up_sampling2d/Shape?
,sequential/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/up_sampling2d/strided_slice/stack?
.sequential/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/up_sampling2d/strided_slice/stack_1?
.sequential/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/up_sampling2d/strided_slice/stack_2?
&sequential/up_sampling2d/strided_sliceStridedSlice'sequential/up_sampling2d/Shape:output:05sequential/up_sampling2d/strided_slice/stack:output:07sequential/up_sampling2d/strided_slice/stack_1:output:07sequential/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2(
&sequential/up_sampling2d/strided_slice?
sequential/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
sequential/up_sampling2d/Const?
sequential/up_sampling2d/mulMul/sequential/up_sampling2d/strided_slice:output:0'sequential/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
sequential/up_sampling2d/mul?
5sequential/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborAsequential/resizing/resize/ResizeNearestNeighbor:resized_images:0 sequential/up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(27
5sequential/up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentityFsequential/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
E
)__inference_sequential_layer_call_fn_6369

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
GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63222
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
L
/__inference_nearest_reference_layer_call_fn_662
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
GPU2*0J 8? *S
fNRL
J__inference_nearest_reference_layer_call_and_return_conditional_losses_6422
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

h
D__inference_sequential_layer_call_and_return_conditional_losses_6313
resizing_input
identity?
resizing/PartitionedCallPartitionedCallresizing_input*
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
GPU2*0J 8? *K
fFRD
B__inference_resizing_layer_call_and_return_conditional_losses_62972
resizing/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_62812
up_sampling2d/PartitionedCall?
IdentityIdentity&up_sampling2d/PartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:z v
J
_output_shapes8
6:4????????????????????????????????????
(
_user_specified_nameresizing_input
?	
`
D__inference_sequential_layer_call_and_return_conditional_losses_6322

inputs
identity?
resizing/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_resizing_layer_call_and_return_conditional_losses_62972
resizing/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_62812
up_sampling2d/PartitionedCall?
IdentityIdentity&up_sampling2d/PartitionedCall:output:0*
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
`
D__inference_sequential_layer_call_and_return_conditional_losses_6364

inputs
identity}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing/resize/size?
%resizing/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2'
%resizing/resize/ResizeNearestNeighbor?
up_sampling2d/ShapeShape6resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor6resizing/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
?
_
C__inference_sequential_layer_call_and_return_conditional_losses_632

inputs
identity}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing/resize/size?
%resizing/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2'
%resizing/resize/ResizeNearestNeighbor?
up_sampling2d/ShapeShape6resizing/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor6resizing/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d/mul:z:0*
T0*:
_output_shapes(
&:$????????????????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
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
,__inference_up_sampling2d_layer_call_fn_6287

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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_62812
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
?

h
D__inference_sequential_layer_call_and_return_conditional_losses_6307
resizing_input
identity?
resizing/PartitionedCallPartitionedCallresizing_input*
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
GPU2*0J 8? *K
fFRD
B__inference_resizing_layer_call_and_return_conditional_losses_62972
resizing/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_62812
up_sampling2d/PartitionedCall?
IdentityIdentity&up_sampling2d/PartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:z v
J
_output_shapes8
6:4????????????????????????????????????
(
_user_specified_nameresizing_input
?
g
J__inference_nearest_reference_layer_call_and_return_conditional_losses_612
input_1
identity?
sequential/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_6072
sequential/PartitionedCall?
IdentityIdentity#sequential/PartitionedCall:output:0*
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
/__inference_nearest_reference_layer_call_fn_657
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
GPU2*0J 8? *S
fNRL
J__inference_nearest_reference_layer_call_and_return_conditional_losses_6422
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
_user_specified_namex"?J
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
PartitionedCall:0$????????????????????tensorflow/serving/predict:?v
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
_tf_keras_model?{"class_name": "NearestReference", "name": "nearest_reference", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "NearestReference"}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_input"}}, {"class_name": "Resizing", "config": {"name": "resizing", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_input"}}, {"class_name": "Resizing", "config": {"name": "resizing", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}}]}}}
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
_tf_keras_layer?{"class_name": "Resizing", "name": "resizing", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resizing", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}
?
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
__inference__wrapped_model_6261?
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
/__inference_nearest_reference_layer_call_fn_647
/__inference_nearest_reference_layer_call_fn_662
/__inference_nearest_reference_layer_call_fn_652
/__inference_nearest_reference_layer_call_fn_657?
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
J__inference_nearest_reference_layer_call_and_return_conditional_losses_506
J__inference_nearest_reference_layer_call_and_return_conditional_losses_612
J__inference_nearest_reference_layer_call_and_return_conditional_losses_637
J__inference_nearest_reference_layer_call_and_return_conditional_losses_443?
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
)__inference_sequential_layer_call_fn_6336
)__inference_sequential_layer_call_fn_6369
)__inference_sequential_layer_call_fn_6325
)__inference_sequential_layer_call_fn_6374?
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
D__inference_sequential_layer_call_and_return_conditional_losses_6350
D__inference_sequential_layer_call_and_return_conditional_losses_6307
D__inference_sequential_layer_call_and_return_conditional_losses_6313
D__inference_sequential_layer_call_and_return_conditional_losses_6364?
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
+B)
"__inference_signature_wrapper_6268x
?2?
'__inference_resizing_layer_call_fn_6385?
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
B__inference_resizing_layer_call_and_return_conditional_losses_6380?
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
,__inference_up_sampling2d_layer_call_fn_6287?
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
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6281?
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
__inference__wrapped_model_6261?M?J
C?@
>?;
x4????????????????????????????????????
? "F?C
A
output_15?2
output_1$?????????????????????
J__inference_nearest_reference_layer_call_and_return_conditional_losses_443?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
J__inference_nearest_reference_layer_call_and_return_conditional_losses_506?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
J__inference_nearest_reference_layer_call_and_return_conditional_losses_612?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
J__inference_nearest_reference_layer_call_and_return_conditional_losses_637?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
/__inference_nearest_reference_layer_call_fn_647?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "+?($?????????????????????
/__inference_nearest_reference_layer_call_fn_652?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "+?($?????????????????????
/__inference_nearest_reference_layer_call_fn_657?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "+?($?????????????????????
/__inference_nearest_reference_layer_call_fn_662?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "+?($?????????????????????
B__inference_resizing_layer_call_and_return_conditional_losses_6380?R?O
H?E
C?@
inputs4????????????????????????????????????
? "6?3
,?)
0"?????????88?????????
? ?
'__inference_resizing_layer_call_fn_6385R?O
H?E
C?@
inputs4????????????????????????????????????
? ")?&"?????????88??????????
D__inference_sequential_layer_call_and_return_conditional_losses_6307?b?_
X?U
K?H
resizing_input4????????????????????????????????????
p

 
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_6313?b?_
X?U
K?H
resizing_input4????????????????????????????????????
p 

 
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_6350?Z?W
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
D__inference_sequential_layer_call_and_return_conditional_losses_6364?Z?W
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
)__inference_sequential_layer_call_fn_6325?b?_
X?U
K?H
resizing_input4????????????????????????????????????
p

 
? ";?84?????????????????????????????????????
)__inference_sequential_layer_call_fn_6336?b?_
X?U
K?H
resizing_input4????????????????????????????????????
p 

 
? ";?84?????????????????????????????????????
)__inference_sequential_layer_call_fn_6369?Z?W
P?M
C?@
inputs4????????????????????????????????????
p

 
? ";?84?????????????????????????????????????
)__inference_sequential_layer_call_fn_6374?Z?W
P?M
C?@
inputs4????????????????????????????????????
p 

 
? ";?84?????????????????????????????????????
"__inference_signature_wrapper_6268?R?O
? 
H?E
C
x>?;
x4????????????????????????????????????"F?C
A
output_15?2
output_1$?????????????????????
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6281?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_up_sampling2d_layer_call_fn_6287?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????