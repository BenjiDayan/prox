from benji_prox_dataloader import *
from simple_transformer import PoseTransformer
import tqdm
import json
from visualisation import predict_and_visualise_transformer

batch_size = 15
in_frames = 15
pred_frames = 30
frame_jump = 10
window_overlap_factor = 8
lr = 0.0001
n_iter = 500
save_every = 40
num_workers = 0


root_dir = "../data_valid/"
smplx_model_path = 'C:\\Users\\xiyi\\projects\\semester_project\\smplify-x\\smplx_model\\models\\'
viz_folder = '../viz_prox_validation/'
save_folder = 'saves'
val_areas = ['N3OpenArea']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd_valid = proxDatasetSkeleton(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                             output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=frame_jump,
                             window_overlap_factor=window_overlap_factor, extra_prefix='joints_worldnorm.pkl')

pd_valid.sequences = [seq for seq in pd_valid.sequences if any([area in seq[0] for area in val_areas])]


# pdc = DatasetBase(root_dir='../data_train/recordings', in_frames=in_frames, pred_frames=pred_frames,
#                   search_prefix='Color', extra_prefix='', frame_jump=frame_jump,
#                   window_overlap_factor=window_overlap_factor)
#
# pdc.align(pd_valid)
# pd_valid.align(pdc)

model = PoseTransformer(num_tokens=25*3).to(device)
model.load_state_dict(torch.load(os.path.join(save_folder, 'transformer_best_model.pt'))['model_state_dict'])
model.eval()

for idx in tqdm.tqdm(range(len(pd_valid))):
    # idx = np.random.randint(pd_valid.bounds[-1])
    try:
        (_, (in_frames_fns_skeletons, in_skels), (pred_frames_fns_skeleton, fut_skels)) = pd_valid.__getitem__(idx)
        in_frames_fns = []
        pred_frames_fns = []
        for in_frame_fn_skeletons in in_frames_fns_skeletons:
            video_name = in_frame_fn_skeletons.split('\\')[3]
            frame_name = in_frame_fn_skeletons.split('\\')[-2]
            in_frames_fns.append('../data_train/recordings/' + video_name + '/Color/' + frame_name + '.jpg')

        for pred_frame_fn_skeletons in pred_frames_fns_skeleton:
            video_name = pred_frame_fn_skeletons.split('\\')[3]
            frame_name = pred_frame_fn_skeletons.split('\\')[-2]
            pred_frames_fns.append('../data_train/recordings/' + video_name + '/Color/' + frame_name + '.jpg')
        # _, in_frames_fns, _, pred_frames_fns = pdc.__getitem__(idx)
        in_imgs = [np.array(cv2.imread(fn)) for fn in in_frames_fns]
        fut_imgs = [np.array(cv2.imread(fn)) for fn in pred_frames_fns]
    except Exception as e:  # some skel fn None so get idx None, None. Or image files unreadable etc.
        continue
    if in_frames_fns == [] or pred_frames_fns == []:
        continue

    if in_skels is not None and fut_skels is not None:
        in_skels_world = torch.cat(in_skels)
        fut_skels_world = torch.cat(fut_skels)
    if torch.any(torch.isnan(in_skels_world)) or torch.any(torch.isnan(fut_skels_world)):
        continue

    scene_name = video_name.split('_')[0]


    # if os.path.exists(os.path.join(viz_folder, video_name)):
    #     continue

    with open(f'{root_dir}/cam2world/{scene_name}.json') as file:
        cam2world = np.array(json.load(file))
        cam2world = torch.from_numpy(cam2world).float()

    images = in_imgs + fut_imgs
    img_fns = in_frames_fns + pred_frames_fns

    output_images = predict_and_visualise_transformer(model, in_skels_world.to(device), fut_skels_world.to(device), images, cam2world.to(device))
    images_down = [
        cv2.resize((img * 255).astype(np.uint8), dsize=(int(img.shape[1] / 5), int(img.shape[0] / 5))) for img
        in output_images]
    if not os.path.exists(os.path.join(viz_folder, video_name)):
        os.makedirs(os.path.join(viz_folder, video_name))
    if not os.path.exists(os.path.join(viz_folder, video_name, 'seq_' + str(idx))):
        os.makedirs(os.path.join(viz_folder, video_name, 'seq_' + str(idx)))
    for i, img in enumerate(images_down):
        cv2.imwrite(os.path.join(viz_folder, video_name, 'seq_' + str(idx), img_fns[i].split('/')[-1]), img)
