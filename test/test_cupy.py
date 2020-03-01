import cupy
import src.pyicecake as pyic

if __name__=="__main__":
    array1 = cupy.array([0, 1, 2], dtype=cupy.float32)
    dltensor = array1.toDlpack()
    gc = pyic.GPUCache(4*1024*1024)
    gc.put_dltensor("tensor1", dltensor)
    dltensor2 = gc.get_dltensor("tensor1", 0)
    array2 = cupy.fromDlpack(dltensor2)
    print(array1)
    print(array2)
    cupy.testing.assert_array_equal(array1, array2)
    print("ok")
