import reedsolo
import bitarray

def encode_bits(bita):
    """
    Takes a bitarray as input and outputs a bitarray
    Bitarray length must be multiple of 8 I think
    Otherwise unspecified results
    (Takes 2400 bits and pads it with 32*8 bits) (for 16 correctable bytes)
    (I think this can be more efficient...I can implement our own bytes)
    """
    bytea = bytearray(bita)
    rs = reedsolo.RSCodec(32)
    bitar = bitearray.bitearray()
    bitar.frombytes(str(rs.encode(bytea)))
    return bitar

def decode_bits(bita):
    """inverse"""
    bytea = bytearray(bita)
    rs = reedsolo.RSCodec(32)
    bitar = bitearray.bitearray()
    bitar.frombytes(str(rs.decode(bytea)))
    return bitar
