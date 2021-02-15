import numpy as np
import SimpleITK as sitk


def load_dicom_slice(name):

    reader = sitk.ImageFileReader()
    reader.SetFileName(name)
    
    
    img = sitk.GetArrayFromImage(reader.Execute())
    
    img = img.transpose(1,2,0)
    
    
    return img[:,:,0]




