with open('periplasmUniRef50.csv') as f_input_1:
	#with open('cytoplasmUniRef50.csv') as f_input_2
	with open('merge_cyto_periplasm.csv', 'w') as f_output:
		for i in range(2001):
			f_output.write(f_input_1.readline())
		with open('cytoplasmUniRef50.csv') as f_input_2:
			f_input_2.readline()
			for i in range(2000):
				f_output.write(f_input_2.readline())
			
