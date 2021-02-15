import numpy as np
import SimpleITK as sitk
from utils.write_ITK_metaimage import write_ITK_metaimage


def get_size_raw(file_name):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    
    file_reader.ReadImageInformation()
    size = file_reader.GetSize()  
    
    return size
    


def read_raw(file_name,extract_size=None,current_index=None):
    
    
    
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    
    file_reader.ReadImageInformation()
    size = file_reader.GetSize()    
    
    
    if  extract_size == None:
        extract_size = size
        
    if  current_index == None:
        current_index = [0,0,0]    
    
    
    img=np.zeros(extract_size,dtype=np.float32)
    
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    
    tmp=sitk.GetArrayFromImage(file_reader.Execute())

    img = tmp.transpose(2,1,0)
    
    
    return img


def write_raw(volume,name):
    
    volume = np.transpose(volume,[2, 1, 0])
    write_ITK_metaimage(volume, name[:-4])
    