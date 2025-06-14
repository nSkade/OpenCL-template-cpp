__kernel void Hello(__global char* data)
{
	int id = get_global_id(0);
	char message[13] = "Hello OCL!\n";
	
	for(int i = 0; i < 13; i++) {
		data[id * 13 + i] = message[i];
	}
}
