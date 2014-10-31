"""Useful functions for working with genetic algorithms."""

def binary_to_int(binary_list, offset):
    """
    Takes a list of binary values, returns a integer representation

    offset is the lowest value that can be return (if binary list is all 0s)
    the maximum value is determined by the number of bits in binary_list
    """

    #convert the binary to an integer
    binary_list = [str(bit) for bit in binary_list] #convert values in binary list to strings
    binary_string = ''.join(binary_list) #convert the list of binary values into a string
    integer = int(binary_string, 2) #convert the string into an integer

    return integer+offset

def binary_to_float(binary_list, minimum, maximum):
    """
    Takes a list of binary values, returns a float representation

    minimum is the lowest value that can be return (if binary list is all 0s)
    maximum is the highest value that can be returned (if binary list is all 1s)
    """
    #get the max value
    max_binary = 2**len(binary_list)-1

    #convert the binary to an integer
    integer = binary_to_int(binary_list, 0)

    #convert the integer to a floating point 
    floating_point = float(integer)/max_binary

    #scale the floating point from min to max
    scaled_floating_point = floating_point*maximum
    scaled_floating_point -= floating_point*minimum
    scaled_floating_point += minimum

    return scaled_floating_point