import torch

scalar = torch.tensor(7) 
print(scalar.ndim)
scalar.item() #covert type to python integer - Get the Python number within a tensor 
scalar = scalar.item()
print(scalar)

random_tensor = torch.rand(size =(3,4)) #create a random tensor of zise 3,4
# print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size = (224, 224, 3)) #create a random image size tensor
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

################# ----- Zero's and Ones ############################

zeros = torch.zeros(size =(3,4)) # Create a tensor of all zeros
print(zeros, zeros.dtype)

ones = torch.ones(size = (3,4)) # Create a tensor of all ones
print(ones, ones.dtype)

############### --- Create a range and tensors like --- ##########

zero_to_ten = torch.arange(start=0, end=10, step=1) # Create a range of values 0 to 10
print(zero_to_ten)

ten_zeros = torch.zeros_like(zero_to_ten) # Can also create a tensor of zeros similar to another tensor (copies shape of specified input)
print(ten_zeros)

############### --- Tensor datatypes --- ##########
float_32_tensor = torch.tensor([3.0 , 6.0, 9.0], 
                               dtype = None,
                               device= None,
                               requires_grad=False)

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

############### ---Manipulating tensors --- ##########
tensor = torch.tensor([1, 2, 3])
print(tensor+10 ) # add it by 10
print(tensor*10) # Multiply it by 10
tensor = tensor -10
print(tensor)
tensor = tensor + 10
print(tensor)

element_wise_mult = tensor* tensor # Element-wise multiplication 
print(element_wise_mult)

############### --- Matrix Multiplication --- ##########

print(tensor.shape)
print(tensor.matmul(tensor))  #matrix multiplication
print(torch.matmul(tensor, tensor))


############### --- Reshaping, stakcing, squeezing and unsqueezing --- ##########
'''
Note - variety of ways to change the dimensions of tensors : Consult documentation
- imporant operations because deep learning models are all about manipulating tensors in some way

'''

x= torch.arange(1., 8.)
print(x, x.shape, x.ndim)

x_reshaped = x.reshape(1,7)  # ?
print(x_reshaped, x_reshaped.shape, x_reshaped.ndim)

z = x.view(1, 7) # Change view- ?
print(z, z.shape)

z[:, 0] = 5 #  changing the view changes the original tensor too
print(z,x)

x_stacked = torch.stack([x,x, x, x],dim=0) # to stack our new tensor on top of itself four times
print(x_stacked)

print(f"Previous tensor: {x_reshaped}")  #  removing all single dimensions from a tensor?
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")


print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

x_unsqueezed  = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# Create tensor with specific shape
x_original = torch.rand(size = (224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted  = x_original.permute(2, 0, 1)  # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape (not permuted): {x_original.shape}")
print(f"New shape (permuted): {x_permuted.shape}")

########## --- indexing (selecting data from tensors) --- #########

x_idx = torch.arange(1, 10).reshape(1, 3, 3)  # takes tensor (vector) and reshapes to a matrix
print(x_idx, x_idx.shape)

#index bracket by bracket -??
print(f"\nFirst square bracket: \n{x_idx[0]}")
print(f"Second square bracket:{x_idx[0][0]}")
print(f"Third square bracket: {x_idx[0][0][0]}")


########## --- PyTorch tensors & NumPy --- #########
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) #Numpy array -> Pytorch tensor
print(f"\nConvert from numpy array to tensor: \n{array, tensor}")

# Tensor to NumPy array
tensor = torch.ones(7)  # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy()
print(f"\nConvert from tensor to numpy array: \n{tensor, numpy_tensor}")







