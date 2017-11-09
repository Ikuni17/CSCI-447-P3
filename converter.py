import sys

def main():
	input('THIS WILL OVERWRITE "converted.arff" in this directory.\nPress enter to continue:')
	try:
		input_file_path = sys.argv[1]
		input_file = open(input_file_path, 'r')
		open('converted.arff', 'w')
		output_file = open('converted.arff', 'a')

		##count how many attributes based on commas
		line = input_file.readline()
		num_attributes = line.count(',') + 1

		##header
		output_file.write('@RELATION ')
		output_file.write(input('Please enter the relation name: ') + '\n')

		# get names and types for each attribute
		for i in range(num_attributes):
			attribute_name = input('Please enter attribute name (' + str(i+1) + '/' + str(num_attributes) +'): ')
			attribute_type = input('Please enter the data type for "' + str(attribute_name) + '": ')

			##    shortcuts
			if attribute_type == 'nu':
				attribute_type = 'NUMERIC'
			elif attribute_type == 'no':
				attribute_type = '{' + input('Please enter possible cases: ') + '}'
			elif attribute_type == 'st':
				attribute_type = 'STRING'
			elif attribute_type == 'da':
				attribute_type = 'DATE'

			output_file.write('\n@ATTRIBUTE ' + str(attribute_name) + ' ' + str(attribute_type) )

		##copy over data from input file
		output_file.write('\n\n@DATA\n' + line)
		for line in input_file:
			output_file.write(line)

	except:
		print('please provide a valid input file')

if __name__ == '__main__':
	main()
