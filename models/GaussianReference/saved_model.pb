??
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
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
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
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?

pipeline
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
l
layer-0
	layer-1

	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
 
?
metrics
non_trainable_variables
	variables
layer_regularization_losses
regularization_losses
trainable_variables

layers
layer_metrics
 
{
_inbound_nodes
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
f
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
?
metrics
non_trainable_variables

	variables
 layer_regularization_losses
regularization_losses
trainable_variables

!layers
"layer_metrics

#0
$1
 
 

0
 
 
 
 
 
 
?
%metrics
&non_trainable_variables
	variables
'layer_regularization_losses
regularization_losses
trainable_variables

(layers
)layer_metrics
 
 
 
 
?
*metrics
+non_trainable_variables
	variables
,layer_regularization_losses
regularization_losses
trainable_variables

-layers
.layer_metrics
 
 
 

0
	1
 
4
	/total
	0count
1	variables
2	keras_api
D
	3total
	4count
5
_fn_kwargs
6	variables
7	keras_api
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
/0
01

1	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

6	variables
?
serving_default_input_1Placeholder*J
_output_shapes8
6:4????????????????????????????????????*
dtype0*?
shape6:4????????????????????????????????????
?
PartitionedCallPartitionedCallserving_default_input_1*
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
#__inference_signature_wrapper_24055
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
__inference__traced_save_24253
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
!__inference__traced_restore_24275ù
?
I
2__inference_gaussian_reference_layer_call_fn_24094
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
GPU2*0J 8? *V
fQRO
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_240402
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
?
d
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24089
x
identity?
#sequential_3/resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2%
#sequential_3/resizing_4/resize/size?
4sequential_3/resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborx,sequential_3/resizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(26
4sequential_3/resizing_4/resize/ResizeNearestNeighbor?
#sequential_3/resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2%
#sequential_3/resizing_5/resize/size?
#sequential_3/resizing_5/resize/CastCast,sequential_3/resizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2%
#sequential_3/resizing_5/resize/Cast?
$sequential_3/resizing_5/resize/ShapeShapeEsequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2&
$sequential_3/resizing_5/resize/Shape?
2sequential_3/resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_3/resizing_5/resize/strided_slice/stack?
4sequential_3/resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_3/resizing_5/resize/strided_slice/stack_1?
4sequential_3/resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_3/resizing_5/resize/strided_slice/stack_2?
,sequential_3/resizing_5/resize/strided_sliceStridedSlice-sequential_3/resizing_5/resize/Shape:output:0;sequential_3/resizing_5/resize/strided_slice/stack:output:0=sequential_3/resizing_5/resize/strided_slice/stack_1:output:0=sequential_3/resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,sequential_3/resizing_5/resize/strided_slice?
%sequential_3/resizing_5/resize/Cast_1Cast5sequential_3/resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2'
%sequential_3/resizing_5/resize/Cast_1?
&sequential_3/resizing_5/resize/truedivRealDiv'sequential_3/resizing_5/resize/Cast:y:0)sequential_3/resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2(
&sequential_3/resizing_5/resize/truediv?
$sequential_3/resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2&
$sequential_3/resizing_5/resize/zeros?
0sequential_3/resizing_5/resize/ScaleAndTranslateScaleAndTranslateEsequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0,sequential_3/resizing_5/resize/size:output:0*sequential_3/resizing_5/resize/truediv:z:0-sequential_3/resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian22
0sequential_3/resizing_5/resize/ScaleAndTranslate?
IdentityIdentityAsequential_3/resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
?
a
E__inference_resizing_5_layer_call_and_return_conditional_losses_24213

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resize/sizel
resize/CastCastresize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resize/CastR
resize/ShapeShapeinputs*
T0*
_output_shapes
:2
resize/Shape?
resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack?
resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack_1?
resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack_2?
resize/strided_sliceStridedSliceresize/Shape:output:0#resize/strided_slice/stack:output:0%resize/strided_slice/stack_1:output:0%resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize/strided_slicey
resize/Cast_1Castresize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resize/Cast_1t
resize/truedivRealDivresize/Cast:y:0resize/Cast_1:y:0*
T0*
_output_shapes
:2
resize/truedivi
resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resize/zeros?
resize/ScaleAndTranslateScaleAndTranslateinputsresize/size:output:0resize/truediv:z:0resize/zeros:output:0*
T0*1
_output_shapes
:???????????*
	antialias( *
kernel_type
gaussian2
resize/ScaleAndTranslate?
IdentityIdentity)resize/ScaleAndTranslate:resized_images:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
R
,__inference_sequential_3_layer_call_fn_23977
resizing_4_input
identity?
PartitionedCallPartitionedCallresizing_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:c _
1
_output_shapes
:???????????
*
_user_specified_nameresizing_4_input
?
H
,__inference_sequential_3_layer_call_fn_24182

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239632
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
I
2__inference_gaussian_reference_layer_call_fn_24099
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
GPU2*0J 8? *V
fQRO
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_240402
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
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_23997

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
?
@
#__inference_signature_wrapper_24055
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
GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_239062
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
O
2__inference_gaussian_reference_layer_call_fn_24048
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
GPU2*0J 8? *V
fQRO
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_240402
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
?
d
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24072
x
identity?
#sequential_3/resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2%
#sequential_3/resizing_4/resize/size?
4sequential_3/resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborx,sequential_3/resizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(26
4sequential_3/resizing_4/resize/ResizeNearestNeighbor?
#sequential_3/resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2%
#sequential_3/resizing_5/resize/size?
#sequential_3/resizing_5/resize/CastCast,sequential_3/resizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2%
#sequential_3/resizing_5/resize/Cast?
$sequential_3/resizing_5/resize/ShapeShapeEsequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2&
$sequential_3/resizing_5/resize/Shape?
2sequential_3/resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_3/resizing_5/resize/strided_slice/stack?
4sequential_3/resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_3/resizing_5/resize/strided_slice/stack_1?
4sequential_3/resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_3/resizing_5/resize/strided_slice/stack_2?
,sequential_3/resizing_5/resize/strided_sliceStridedSlice-sequential_3/resizing_5/resize/Shape:output:0;sequential_3/resizing_5/resize/strided_slice/stack:output:0=sequential_3/resizing_5/resize/strided_slice/stack_1:output:0=sequential_3/resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,sequential_3/resizing_5/resize/strided_slice?
%sequential_3/resizing_5/resize/Cast_1Cast5sequential_3/resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2'
%sequential_3/resizing_5/resize/Cast_1?
&sequential_3/resizing_5/resize/truedivRealDiv'sequential_3/resizing_5/resize/Cast:y:0)sequential_3/resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2(
&sequential_3/resizing_5/resize/truediv?
$sequential_3/resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2&
$sequential_3/resizing_5/resize/zeros?
0sequential_3/resizing_5/resize/ScaleAndTranslateScaleAndTranslateEsequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0,sequential_3/resizing_5/resize/size:output:0*sequential_3/resizing_5/resize/truediv:z:0-sequential_3/resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian22
0sequential_3/resizing_5/resize/ScaleAndTranslate?
IdentityIdentityAsequential_3/resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
R
,__inference_sequential_3_layer_call_fn_23966
resizing_4_input
identity?
PartitionedCallPartitionedCallresizing_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239632
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:c _
1
_output_shapes
:???????????
*
_user_specified_nameresizing_4_input
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_24160

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*1
_output_shapes
:???????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_resizing_4_layer_call_and_return_conditional_losses_23916

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/size?
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_resizing_4_layer_call_fn_24198

inputs
identity?
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
GPU2*0J 8? *N
fIRG
E__inference_resizing_4_layer_call_and_return_conditional_losses_239162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
m
G__inference_sequential_3_layer_call_and_return_conditional_losses_23948
resizing_4_input
identity?
resizing_4/PartitionedCallPartitionedCallresizing_4_input*
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
GPU2*0J 8? *N
fIRG
E__inference_resizing_4_layer_call_and_return_conditional_losses_239162
resizing_4/PartitionedCall?
resizing_5/PartitionedCallPartitionedCall#resizing_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_5_layer_call_and_return_conditional_losses_239392
resizing_5/PartitionedCall?
IdentityIdentity#resizing_5/PartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:c _
1
_output_shapes
:???????????
*
_user_specified_nameresizing_4_input
?
H
,__inference_sequential_3_layer_call_fn_24187

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_sequential_3_layer_call_fn_24138

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239972
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_23954
resizing_4_input
identity?
resizing_4/PartitionedCallPartitionedCallresizing_4_input*
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
GPU2*0J 8? *N
fIRG
E__inference_resizing_4_layer_call_and_return_conditional_losses_239162
resizing_4/PartitionedCall?
resizing_5/PartitionedCallPartitionedCall#resizing_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_5_layer_call_and_return_conditional_losses_239392
resizing_5/PartitionedCall?
IdentityIdentity#resizing_5/PartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:c _
1
_output_shapes
:???????????
*
_user_specified_nameresizing_4_input
? 
=
 __inference__wrapped_model_23906
input_1
identity?
6gaussian_reference/sequential_3/resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   28
6gaussian_reference/sequential_3/resizing_4/resize/size?
Ggaussian_reference/sequential_3/resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinput_1?gaussian_reference/sequential_3/resizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2I
Ggaussian_reference/sequential_3/resizing_4/resize/ResizeNearestNeighbor?
6gaussian_reference/sequential_3/resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   28
6gaussian_reference/sequential_3/resizing_5/resize/size?
6gaussian_reference/sequential_3/resizing_5/resize/CastCast?gaussian_reference/sequential_3/resizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:28
6gaussian_reference/sequential_3/resizing_5/resize/Cast?
7gaussian_reference/sequential_3/resizing_5/resize/ShapeShapeXgaussian_reference/sequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:29
7gaussian_reference/sequential_3/resizing_5/resize/Shape?
Egaussian_reference/sequential_3/resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
Egaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack?
Ggaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Ggaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_1?
Ggaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Ggaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_2?
?gaussian_reference/sequential_3/resizing_5/resize/strided_sliceStridedSlice@gaussian_reference/sequential_3/resizing_5/resize/Shape:output:0Ngaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack:output:0Pgaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_1:output:0Pgaussian_reference/sequential_3/resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2A
?gaussian_reference/sequential_3/resizing_5/resize/strided_slice?
8gaussian_reference/sequential_3/resizing_5/resize/Cast_1CastHgaussian_reference/sequential_3/resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2:
8gaussian_reference/sequential_3/resizing_5/resize/Cast_1?
9gaussian_reference/sequential_3/resizing_5/resize/truedivRealDiv:gaussian_reference/sequential_3/resizing_5/resize/Cast:y:0<gaussian_reference/sequential_3/resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2;
9gaussian_reference/sequential_3/resizing_5/resize/truediv?
7gaussian_reference/sequential_3/resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    29
7gaussian_reference/sequential_3/resizing_5/resize/zeros?
Cgaussian_reference/sequential_3/resizing_5/resize/ScaleAndTranslateScaleAndTranslateXgaussian_reference/sequential_3/resizing_4/resize/ResizeNearestNeighbor:resized_images:0?gaussian_reference/sequential_3/resizing_5/resize/size:output:0=gaussian_reference/sequential_3/resizing_5/resize/truediv:z:0@gaussian_reference/sequential_3/resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian2E
Cgaussian_reference/sequential_3/resizing_5/resize/ScaleAndTranslate?
IdentityIdentityTgaussian_reference/sequential_3/resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
d
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24040
x
identity?
sequential_3/PartitionedCallPartitionedCallx*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_240142
sequential_3/PartitionedCall?
IdentityIdentity%sequential_3/PartitionedCall:output:0*
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
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_24014

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_24116

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
j
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24032
input_1
identity?
sequential_3/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_240142
sequential_3/PartitionedCall?
IdentityIdentity%sequential_3/PartitionedCall:output:0*
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
?
a
E__inference_resizing_5_layer_call_and_return_conditional_losses_23939

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resize/sizel
resize/CastCastresize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resize/CastR
resize/ShapeShapeinputs*
T0*
_output_shapes
:2
resize/Shape?
resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack?
resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack_1?
resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
resize/strided_slice/stack_2?
resize/strided_sliceStridedSliceresize/Shape:output:0#resize/strided_slice/stack:output:0%resize/strided_slice/stack_1:output:0%resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize/strided_slicey
resize/Cast_1Castresize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resize/Cast_1t
resize/truedivRealDivresize/Cast:y:0resize/Cast_1:y:0*
T0*
_output_shapes
:2
resize/truedivi
resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resize/zeros?
resize/ScaleAndTranslateScaleAndTranslateinputsresize/size:output:0resize/truediv:z:0resize/zeros:output:0*
T0*1
_output_shapes
:???????????*
	antialias( *
kernel_type
gaussian2
resize/ScaleAndTranslate?
IdentityIdentity)resize/ScaleAndTranslate:resized_images:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?	
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_23974

inputs
identity?
resizing_4/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *N
fIRG
E__inference_resizing_4_layer_call_and_return_conditional_losses_239162
resizing_4/PartitionedCall?
resizing_5/PartitionedCallPartitionedCall#resizing_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_5_layer_call_and_return_conditional_losses_239392
resizing_5/PartitionedCall?
IdentityIdentity#resizing_5/PartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_24133

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*8
_output_shapes&
$:"?????????88?????????*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*:
_output_shapes(
&:$????????????????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
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
?
F
*__inference_resizing_5_layer_call_fn_24218

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_5_layer_call_and_return_conditional_losses_239392
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?	
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_23963

inputs
identity?
resizing_4/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *N
fIRG
E__inference_resizing_4_layer_call_and_return_conditional_losses_239162
resizing_4/PartitionedCall?
resizing_5/PartitionedCallPartitionedCall#resizing_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_resizing_5_layer_call_and_return_conditional_losses_239392
resizing_5/PartitionedCall?
IdentityIdentity#resizing_5/PartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_24253
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
value3B1 B+_temp_86f2df2964e140acba564edc3dfcd1bf/part2	
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
j
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24027
input_1
identity?
sequential_3/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_239972
sequential_3/PartitionedCall?
IdentityIdentity%sequential_3/PartitionedCall:output:0*
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
?
?
!__inference__traced_restore_24275
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
?
c
G__inference_sequential_3_layer_call_and_return_conditional_losses_24177

inputs
identity?
resizing_4/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resizing_4/resize/size?
'resizing_4/resize/ResizeNearestNeighborResizeNearestNeighborinputsresizing_4/resize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2)
'resizing_4/resize/ResizeNearestNeighbor?
resizing_5/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
resizing_5/resize/size?
resizing_5/resize/CastCastresizing_5/resize/size:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast?
resizing_5/resize/ShapeShape8resizing_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
resizing_5/resize/Shape?
%resizing_5/resize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%resizing_5/resize/strided_slice/stack?
'resizing_5/resize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_1?
'resizing_5/resize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'resizing_5/resize/strided_slice/stack_2?
resizing_5/resize/strided_sliceStridedSlice resizing_5/resize/Shape:output:0.resizing_5/resize/strided_slice/stack:output:00resizing_5/resize/strided_slice/stack_1:output:00resizing_5/resize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2!
resizing_5/resize/strided_slice?
resizing_5/resize/Cast_1Cast(resizing_5/resize/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:2
resizing_5/resize/Cast_1?
resizing_5/resize/truedivRealDivresizing_5/resize/Cast:y:0resizing_5/resize/Cast_1:y:0*
T0*
_output_shapes
:2
resizing_5/resize/truediv
resizing_5/resize/zerosConst*
_output_shapes
:*
dtype0*
valueB*    2
resizing_5/resize/zeros?
#resizing_5/resize/ScaleAndTranslateScaleAndTranslate8resizing_4/resize/ResizeNearestNeighbor:resized_images:0resizing_5/resize/size:output:0resizing_5/resize/truediv:z:0 resizing_5/resize/zeros:output:0*
T0*1
_output_shapes
:???????????*
	antialias( *
kernel_type
gaussian2%
#resizing_5/resize/ScaleAndTranslate?
IdentityIdentity4resizing_5/resize/ScaleAndTranslate:resized_images:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_sequential_3_layer_call_fn_24143

inputs
identity?
PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_240142
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
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
?
a
E__inference_resizing_4_layer_call_and_return_conditional_losses_24193

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"8   8   2
resize/size?
resize/ResizeNearestNeighborResizeNearestNeighborinputsresize/size:output:0*
T0*/
_output_shapes
:?????????88*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
O
2__inference_gaussian_reference_layer_call_fn_24043
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
GPU2*0J 8? *V
fQRO
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_240402
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
_user_specified_name	input_1"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
^
input_1S
serving_default_input_1:04????????????????????????????????????G
output_1;
PartitionedCall:0$????????????????????tensorflow/serving/predict:?w
?
pipeline
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
8_default_save_signature
9__call__
*:&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "GaussianReference", "name": "gaussian_reference", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "GaussianReference"}, "training_config": {"loss": "mse", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
layer-0
	layer-1

	variables
regularization_losses
trainable_variables
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_4_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_4", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "Resizing", "config": {"name": "resizing_5", "trainable": true, "dtype": "float32", "height": 224, "width": 224, "interpolation": "gaussian"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_4_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_4", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}, {"class_name": "Resizing", "config": {"name": "resizing_5", "trainable": true, "dtype": "float32", "height": 224, "width": 224, "interpolation": "gaussian"}}]}}}
"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
non_trainable_variables
	variables
layer_regularization_losses
regularization_losses
trainable_variables

layers
layer_metrics
9__call__
8_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
=serving_default"
signature_map
?
_inbound_nodes
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
>__call__
*?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Resizing", "name": "resizing_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resizing_4", "trainable": true, "dtype": "float32", "height": 56, "width": 56, "interpolation": "nearest"}}
?
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Resizing", "name": "resizing_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resizing_5", "trainable": true, "dtype": "float32", "height": 224, "width": 224, "interpolation": "gaussian"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
non_trainable_variables

	variables
 layer_regularization_losses
regularization_losses
trainable_variables

!layers
"layer_metrics
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
%metrics
&non_trainable_variables
	variables
'layer_regularization_losses
regularization_losses
trainable_variables

(layers
)layer_metrics
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*metrics
+non_trainable_variables
	variables
,layer_regularization_losses
regularization_losses
trainable_variables

-layers
.layer_metrics
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	/total
	0count
1	variables
2	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	3total
	4count
5
_fn_kwargs
6	variables
7	keras_api"?
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
/0
01"
trackable_list_wrapper
-
1	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
-
6	variables"
_generic_user_object
?2?
 __inference__wrapped_model_23906?
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
annotations? *I?F
D?A
input_14????????????????????????????????????
?2?
2__inference_gaussian_reference_layer_call_fn_24043
2__inference_gaussian_reference_layer_call_fn_24099
2__inference_gaussian_reference_layer_call_fn_24048
2__inference_gaussian_reference_layer_call_fn_24094?
???
FullArgSpec$
args?
jself
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
annotations? *
 
?2?
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24089
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24072
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24027
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24032?
???
FullArgSpec$
args?
jself
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
annotations? *
 
?2?
,__inference_sequential_3_layer_call_fn_24138
,__inference_sequential_3_layer_call_fn_24182
,__inference_sequential_3_layer_call_fn_24187
,__inference_sequential_3_layer_call_fn_24143
,__inference_sequential_3_layer_call_fn_23977
,__inference_sequential_3_layer_call_fn_23966?
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
?2?
G__inference_sequential_3_layer_call_and_return_conditional_losses_24116
G__inference_sequential_3_layer_call_and_return_conditional_losses_24133
G__inference_sequential_3_layer_call_and_return_conditional_losses_23948
G__inference_sequential_3_layer_call_and_return_conditional_losses_24160
G__inference_sequential_3_layer_call_and_return_conditional_losses_24177
G__inference_sequential_3_layer_call_and_return_conditional_losses_23954?
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
2B0
#__inference_signature_wrapper_24055input_1
?2?
*__inference_resizing_4_layer_call_fn_24198?
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
E__inference_resizing_4_layer_call_and_return_conditional_losses_24193?
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
*__inference_resizing_5_layer_call_fn_24218?
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
E__inference_resizing_5_layer_call_and_return_conditional_losses_24213?
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
 ?
 __inference__wrapped_model_23906?S?P
I?F
D?A
input_14????????????????????????????????????
? "F?C
A
output_15?2
output_1$?????????????????????
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24027?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24032?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24072?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "8?5
.?+
0$????????????????????
? ?
M__inference_gaussian_reference_layer_call_and_return_conditional_losses_24089?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "8?5
.?+
0$????????????????????
? ?
2__inference_gaussian_reference_layer_call_fn_24043?W?T
M?J
D?A
input_14????????????????????????????????????
p
? "+?($?????????????????????
2__inference_gaussian_reference_layer_call_fn_24048?W?T
M?J
D?A
input_14????????????????????????????????????
p 
? "+?($?????????????????????
2__inference_gaussian_reference_layer_call_fn_24094?Q?N
G?D
>?;
x4????????????????????????????????????
p
? "+?($?????????????????????
2__inference_gaussian_reference_layer_call_fn_24099?Q?N
G?D
>?;
x4????????????????????????????????????
p 
? "+?($?????????????????????
E__inference_resizing_4_layer_call_and_return_conditional_losses_24193j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????88
? ?
*__inference_resizing_4_layer_call_fn_24198]9?6
/?,
*?'
inputs???????????
? " ??????????88?
E__inference_resizing_5_layer_call_and_return_conditional_losses_24213j7?4
-?*
(?%
inputs?????????88
? "/?,
%?"
0???????????
? ?
*__inference_resizing_5_layer_call_fn_24218]7?4
-?*
(?%
inputs?????????88
? ""?????????????
G__inference_sequential_3_layer_call_and_return_conditional_losses_23948~K?H
A?>
4?1
resizing_4_input???????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_23954~K?H
A?>
4?1
resizing_4_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_24116?Z?W
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_24133?Z?W
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_24160tA?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_24177tA?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
,__inference_sequential_3_layer_call_fn_23966qK?H
A?>
4?1
resizing_4_input???????????
p

 
? ""?????????????
,__inference_sequential_3_layer_call_fn_23977qK?H
A?>
4?1
resizing_4_input???????????
p 

 
? ""?????????????
,__inference_sequential_3_layer_call_fn_24138?Z?W
P?M
C?@
inputs4????????????????????????????????????
p

 
? "+?($?????????????????????
,__inference_sequential_3_layer_call_fn_24143?Z?W
P?M
C?@
inputs4????????????????????????????????????
p 

 
? "+?($?????????????????????
,__inference_sequential_3_layer_call_fn_24182gA?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
,__inference_sequential_3_layer_call_fn_24187gA?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
#__inference_signature_wrapper_24055?^?[
? 
T?Q
O
input_1D?A
input_14????????????????????????????????????"F?C
A
output_15?2
output_1$????????????????????