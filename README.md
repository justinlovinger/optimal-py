# Optimal (beta)
A python metaheuristic optimization library. Built for easy extension and usage.

Warning: Optimal is in beta. Api may change. I will do my best to note any breaking changes in this readme, but no guarantee is given.

Supported metaheuristics:

* Genetic algorithms (GA)
* Gravitational search algorithm (GSA)
* Cross entropy (CE)

# Installation
Copy the "optimal" folder to [python-path]/lib/site-packages

# Usage
	import math

	from optimal import genalg
	from optimal import helpers

	# The genetic algorithm uses binary solutions.
	# A decode function is useful for converting the binary solution to real numbers
	def decode_ackley(binary):
		# Helpful functions from helpers are used to convert binary to float
		# x1 and x2 range from -5.0 to 5.0
		x1 = helpers.binary_to_float(binary[0:16], -5.0, 5.0)
		x2 = helpers.binary_to_float(binary[16:32], -5.0, 5.0)
		return x1, x2

	# ackley is our fitness function
	# This is how a user defines their problem
	def ackley(solution, decode_func):
		# Turn our solution of bits into floating point values
		x1, x2 = decode_func(solution)

		# Ackley's function
		# A common mathematical optimization problem
		output = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1**2 + x2**2))) - math.exp(
			0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2))) + 20 + math.e

		# You can prematurely stop the metaheuristic by returning True
		# as the second return value
		# Here, we consider the problem solved if the output is <= 0.01
		finished = output <= 0.01

		# Because this function is trying to minimize the output,
		# a smaller output has a greater fitness
		fitness = 1 / output

		# First return argument must be a real number
		# The higher the number, the better the solution
		# Second return argument is a boolean, and optional
		return fitness, finished

	# Setup and run the genetic algorithm, using our fitness function, 
	# and a chromosome size of 32
	# Additional fitness function arguments are added as keyword arguments
	my_genalg = genalg.GenAlg(ackley, 32,
							  decode_func=decode_ackley)
	best_solution = my_genalg.optimize()

	print decode_ackley(best_solution)
	
Important notes:
* Fitness function must take solution as its first argument
* Fitness function must return a real number as its first return value

For further usage details, see comprehensive doc strings.
