?	c?ZB>???c?ZB>???!c?ZB>???	?r????@?r????@!?r????@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$c?ZB>?????	h"??A?C?l????Y??Pk?w??*	?????9W@2F
Iterator::Modelm???{???!+"Q??A@)?{??Pk??1}?????;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ͪ?Ֆ?!^j?-8@)jM????1??:K?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<?R??!Zz??Qw7@)??d?`T??1j]???D3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu????!??}ײP@)F%u???1?M&U?j,@:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ?}?!i'?i?n@)?<,Ԛ?}?1i'?i?n@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?q????o?!?s????@)?q????o?1?s????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6j?!4c?w?@)-C??6j?14c?w?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??+e???!p*??;(:@){?G?zd?1??oM?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?r????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??	h"????	h"??!??	h"??      ??!       "      ??!       *      ??!       2	?C?l?????C?l????!?C?l????:      ??!       B      ??!       J	??Pk?w????Pk?w??!??Pk?w??R      ??!       Z	??Pk?w????Pk?w??!??Pk?w??JCPU_ONLYY?r????@b Y      Y@q??{M@"?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?58.9619% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 