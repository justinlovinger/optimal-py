###############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2014 Justin Lovinger
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################

"""Useful functions for working with genetic algorithms."""

def binary_to_int(binary_list, offset):
    """Takes a list of binary values, returns a integer representation.

   The maximum value is determined by the number of bits in binary_list,
   and the offset.

    Args:
        Offset: The lowest value that can be return (if binary list is all 0s).
    
    Returns:
        int; An integer.   
    """

    #convert the binary to an integer
    binary_list = [str(bit) for bit in binary_list] #convert values in binary list to strings
    binary_string = ''.join(binary_list) #convert the list of binary values into a string
    integer = int(binary_string, 2) #convert the string into an integer

    return integer+offset

def binary_to_float(binary_list, minimum, maximum):
    """Takes a list of binary values, returns a float representation.

    Args:
        minimum: The lowest value that can be return (if binary list is all 0s).
        maximum: The highest value that can be returned (if binary list is all 1s).

    Returns:
        float; A floating point number.
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