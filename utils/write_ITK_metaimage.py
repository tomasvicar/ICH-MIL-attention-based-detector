import numpy
import os

def write_ITK_metaimage(volume, name, spacing=[1,1,1]):
    """
    Writes a ITK metaimage file, which can be viewed by Paraview.
    See http://www.itk.org/Wiki/ITK/MetaIO/Documentation
    Generates a raw file containing the volume, and an associated mhd
    metaimge file.
    TODO: Order should not be given as an argument, but guessed from the
    layout of the numpy array (possibly modified).
    TODO: Use compressed raw binary to save space. Should be possible, but
    given the lack of documentation it is a pain in the ass.
    Parameters
    ----------
    volume : numpy.array
        Input 3D image. Must be numpy.float32
    name : string
        Name of the metaimage file.
    """
    
    order = [2, 1, 0]
    assert len(volume.shape) == 3
    print("* Writing ITK metaimage " + name + "...")
    # Write volume data
    with open(name + ".raw", "wb") as raw_file:
        raw_file.write(bytearray(volume.astype(volume.dtype).flatten()))
    # Compute meta data
    if volume.dtype == numpy.float32:
        typename = 'MET_FLOAT'
    elif volume.dtype == numpy.uint8:
        typename = 'MET_CHAR'
    else:
        raise RuntimeError("Incorrect element type: " + volume.dtype)
    # Write meta data
    with open(name + ".mhd", "w") as meta_file:
        basename = os.path.basename(name)
        meta_file.write("ObjectType = Image\nNDims = 3\n")
        meta_file.write(
            "ElementSpacing = " + str(spacing[order[0]]) + " " +
            str(spacing[order[1]]) + " " +
            str(spacing[order[2]]) + "\n")
        
        meta_file.write(
            "DimSize = " + str(volume.shape[order[0]]) + " " +
            str(volume.shape[order[1]]) + " " +
            str(volume.shape[order[2]]) + "\n")
        
        
        meta_file.write(
            "ElementType = {0}\nElementDataFile = {1}.raw\n".format(
                typename, basename))