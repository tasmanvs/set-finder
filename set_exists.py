# Cards have one of each:
# 	Shape:
# 		Oval
# 		Squiggle
# 		Diamond
# 	Color:
# 		Red
# 		Green
# 		Purple
# 	Shading:
# 		Empty
# 		Solid
# 		Striped
# 	Number:
# 		1
# 		2
# 		3
# For convenience, you could assign numbers to each one. For example: Oval = 1, Squiggle = 2, Diamond = 3.


# Class definition:
class Card(object):
	# constructor for when we want to declare a new card. see myCard = Card(~~~)
	def __init__(self, shape, color, shading, number): 
		self.shape = shape
		self.color = color
		self.shading = shading
		self.number = number

	# Allows us to print the object as a string.
	# Try commenting this out and see what happens when you try to print something.
 	def __str__(self):
 		shape_str = "Shape:    " + str(self.shape)
 		color_str = "Color:    " + str(self.color)
 		shading_str = "Shading:  " + str(self.shading)
 		number_str = "Number:   " + str(self.number) 
 		print_str = shape_str + "\n" + color_str + "\n" + shading_str + "\n" + number_str + "\n"
 		return print_str

 ###############################################################################################

# The cards argument is an array of any number of card objects.
# This function should return an array of sets.
# Each set consists of the three cards that make up that set.
# Sets can overlap, meaning that some cards can be shared between sets.
def find_sets(cards):
	sets = []
	# do the thing
	return sets

# cards is an array of three card objects
def is_a_set(cards):
	# do the thing
	return True #replace this

# should return a card object that is the correct card to complete the set.
def find_third_card(card1, card2):
	correct_card = Card("Oval", "Red", "Striped", 1)
	return correct_card



##########################################################################################

# If you end up assigning numbers to them, it would look something like:
# myCard1 = Card(1,2,1,1)
myCard1 = Card("Oval", "Red", "Striped", 1)
myCard2 = Card("Diamond", "Red", "Filled", 3)
myCard3 = Card("Squiggle", "Red", "Empty", 2)

myCards = [myCard1, myCard2, myCard3] # create an array of card objects

for idx, myCard in enumerate(myCards): # print out each object
	print idx + 1
	print myCard

print "Running find_sets on myCards\n"
found_sets = find_sets(myCards) #this currently does nothing

print "There were", len(found_sets), "sets found.\n"
print "The found_sets are:\n"
for s in found_sets:
	print s

