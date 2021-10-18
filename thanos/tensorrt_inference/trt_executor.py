import ctypes
from typing import Union
import pycuda.autoinit as cuda_init
import pycuda
import tensorrt as trt
from thanos.tensorrt_inference.utils import allocate_buffers, execute_async, execute_sync

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class TRTExecutor:
    """
    A helper class to execute a TensorRT engine.
    Attributes:
    -----------
    stream: pycuda.driver.Stream
    engine: tensorrt.ICudaEngine
    context: tensorrt.IExecutionContext
    inputs/outputs: list[HostDeviceMem]
        see trt_helper.py
    bindings: list[int]
        pointers in GPU for each input/output of the engine
    dict_inputs/dict_outputs: dict[str, HostDeviceMem]
        key = input node name
        value = HostDeviceMem of corresponding binding
    """

    def __init__(
        self,
        engine: Union[str, trt.ICudaEngine],
        has_dynamic_shape: bool = False,
        stream: pycuda.driver.Stream = None,
        sync_mode: bool = False,
        verbose_logger: bool = False,
    ):
        """
        Parameters
        ----------
        engine: if str, path to engine file, if tensorrt.ICudaEngine, serialized engine
        has_dynamic_shape: bool
        stream: pycuda.driver.Stream
            if None, one will be created by allocate_buffers function
        sync_mode: bool, default = False.
            True/False enable the synchronized/asynchonized execution of TensorRT engine
        logger: tensorrt.ILogger, logger to print info in terminal
        """
        self.sync_mode = sync_mode
        self.stream = stream
        if verbose_logger:
            self.logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            self.logger = TRT_LOGGER
        if isinstance(engine, str):
            with open(engine, "rb") as f, trt.Runtime(self.logger) as runtime:
                print("Reading engine  ...")
                self.engine = runtime.deserialize_cuda_engine(f.read())
                assert self.engine is not None, "Read engine failed"
                print("Engine loaded")
        else:
            self.engine = engine
        self.context = self.engine.create_execution_context()
        
        # TODO: test this mode later with DETR segmentaion
        if not has_dynamic_shape:
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
                self.context, self.stream, self.sync_mode
            )
            self.dict_inputs = {mem_obj.name: mem_obj for mem_obj in self.inputs}
            self.dict_outputs = {mem_obj.name: mem_obj for mem_obj in self.outputs}

    def print_bindings_info(self):
        print("ID / Name / isInput / shape / dtype")
        for i in range(self.engine.num_bindings):
            print(
                f"Binding: {i}, name: {self.engine.get_binding_name(i)}, input: {self.engine.binding_is_input(i)}, \
                    shape: {self.engine.get_binding_shape(i)}, dtype: {self.engine.get_binding_dtype(i)}"
            )

    def execute(self):
        if self.sync_mode:
            execute_sync(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs)
        else:
            execute_async(
                self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
            )
        return {out.name: out.host for out in self.outputs}

    def set_binding_shape(self, binding: int, shape: tuple):
        self.context.set_binding_shape(binding, shape)

    def allocate_mem(self):
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.context, self.stream)
        self.dict_inputs = {mem_obj.name: mem_obj for mem_obj in self.inputs}
        self.dict_outputs = {mem_obj.name: mem_obj for mem_obj in self.outputs}

    def __call__(self, *inputs, **kwargs):
        for i, tensor in enumerate(inputs):
            self.inputs[i].host = tensor
        return self.execute()