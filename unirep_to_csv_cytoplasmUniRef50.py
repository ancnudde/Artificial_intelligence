with open('cytoplasmUniRef50.unirep') as f:
	with open('cytoplasmUniRef50.csv', 'w') as f2:
		list_vect = []
		for i in range(1,193):
			list_vect.append('v'+str(i)) 
		f2.write('id,')
		for vect in list_vect:
			f2.write(vect+',')
		f2.write('class\n')
		for i in range(16408):
			f2.write((f.readline()).strip('\n').strip('>')+',')
			f2.write(f.readline().replace(' ', ',').strip('\n')+',')
			f2.write(f.readline().replace(' ', ',').strip('\n')+',')
			f2.write(f.readline().replace(' ', ',').strip('\n')+',')
			f2.write('2\n')
		

