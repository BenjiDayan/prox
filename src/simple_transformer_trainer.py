from benji_prox_dataloader import *
import tqdm
from simple_transformer import PoseTransformer
import torch
from torch import nn

root_dir = "/Users/xiyichen/Documents/virtual_humans/PROXD/"
smplx_model_path='/Users/xiyichen/Documents/semester_project/smplify-x/smplx_model/models/'
in_frames = 10
pred_frames = 5
batch_size = 8
lr = 0.0001
n_iter = 100
# pd = proxDataset(root_dir, in_frames=in_frames, pred_frames=pred_frames, output_type='joint_thetas', smplx_model_path=smplx_model_path)
# dataloader = DataLoader(pd, batch_size=batch_size,
#                         shuffle=True, num_workers=0, collate_fn=my_collate)
pd = proxDataset(root_dir, in_frames=in_frames, pred_frames=pred_frames, output_type='joint_locations', smplx_model_path=smplx_model_path)

dataloader = DataLoader(pd, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=my_collate)
model = PoseTransformer(num_tokens=25*3)
model.train()

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
losses = []
for epoch in range(n_iter):
	total_loss = 0
	for i, (indices, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader))):
		X = in_skels.to(device)
		X_shape = X.shape
		X = X.reshape(X_shape[0], X_shape[1], X_shape[2]*X_shape[3])
		y = fut_skels.to(device)
		y_shape = y.shape
		y = y.reshape(y_shape[0], y_shape[1], y_shape[2]*y_shape[3])
		tgt_mask = model.get_tgt_mask(y_shape[1]).to(device)

		optimizer.zero_grad()
		pred = model(X, y, tgt_mask=tgt_mask)
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()
		total_loss += loss.detach().item()
		if i % 50 == 0:
			print(f'Epoch {epoch}: current mean loss: {total_loss / ((i+1)*batch_size)}')
	print(f'end epoch {epoch}: total mean loss: {total_loss / len(dataloader)}')
