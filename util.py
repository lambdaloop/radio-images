import reedsolo
import bitarray
import zlib
import numpy as np

"""The important functions"""

def compress(npa):
    """takes in a numpy array (length multiple 8), returns it compressed, as a bitarray"""
    bita = bitarray.bitarray(npa.tolist())
    bytea = bytearray(bita)
    compressed = zlib.compress(str(bytea), 9)
    bitar = bitarray.bitarray()
    bitar.frombytes(compressed)
    return bitar

def decompress(npa):
    """takes in a compressed bitarray (length multiple 8), returns bitarray"""
    bytea = bytearray(npa)
    decompressed = zlib.decompress(str(bytea))
    bitar = bitarray.bitarray()
    bitar.frombytes(decompressed)
    return bitar

def encorrect(npa):
    """takes in numpy array (length multiple 8), returns numpy array"""
    bita = bitarray.bitarray(npa.tolist())
    bytea = bytearray(bita)
    rs = reedsolo.RSCodec(4)
    bitar = bitarray.bitarray()
    bitar.frombytes(str(rs.encode(bytea)))
    return np.array(bitar.tolist())

def correct(npa):
    """takes in encorrected numpy array (length multiple 8), returns numpy array"""
    bita = bitarray.bitarray(npa.tolist())
    bytea = bytearray(bita)
    rs = reedsolo.RSCodec(4)
    bitar = bitarray.bitarray()
    try:
        bitar.frombytes(str(rs.decode(bytea)))
    except:
        return npa[:-32]
    return np.array(bitar.tolist())

    

def compress_encorrect(npa):
    """takes in a numpy array (length is multiple of 8),
    compresses it, then error corrects it, and outputs a numpy array"""
    
    bita = bitarray.bitarray(npa.tolist())
    bytea = bytearray(bita)
    compressed = zlib.compress(str(bytea), 9) # change number for compression setting
    bytea = bytearray(compressed)
    rs = reedsolo.RSCodec(4)
    bitar = bitarray.bitarray()
    bitar.frombytes(str(rs.encode(bytea)))
    return np.array(bitar.tolist())

def correct_decompress(npa):
    """takes in a numpy array (length is multiple of 8),
    error corrects it, decompresses it, outputs numpy array"""
    bita = bitarray.bitarray(npa.tolist())
    rs = reedsolo.RSCodec(4)
    decoded = rs.decode(bytearray(bita)) #bytearray
    decompressed = zlib.decompress(str(decoded)) #string
    bitar = bitarray.bitarray()
    bitar.frombytes(decompressed)
    return np.array(bitar.tolist())

"""Extra functions"""

def decode_bits(bita):
    """inverse"""
    bytea = bytearray(bita)
    rs = reedsolo.RSCodec(4)
    bitar = bitarray.bitarray()
    bitar.frombytes(str(rs.decode(bytea)))
    return bitar

def np_to_bits(npa):
    """numpy array to bitarray"""
    return bitarray.bitarray(npa.tolist())

def bits_to_np(bita):
    """bitarray to numpy array"""
    return np.array(bita.tolist())


