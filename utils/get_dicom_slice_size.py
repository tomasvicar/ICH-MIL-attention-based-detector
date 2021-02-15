import numpy as np
import SimpleITK as sitk

def get_dicom_slice_size(name):
    
    
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(name)
    
    reader.ReadImageInformation()
    size = reader.GetSize()
    
    size = size[:2]
    
    return size