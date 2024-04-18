from easydict import EasyDict as edict

configs = edict()

# ---- training
configs['image_dir'] = '/workspace/data/subset'
configs['train_list'] = [
    "bank-statement_new_template-all_text-20230915",
    "bank-statement_new-template_augmented-all_text-20230929",
    "bpkb_page-3-text",
    "bpkb_page-4-text",
    "fidusia_all_texts-20231115-8",
    "ghk_all_text-20231115",
    "kitas-kitap_kitas-cetak",
    "kitas-kitap_kitas-kartu",
    "kk_kk-header-test-230905",
    "kk_kk-header-train-230905",
    "kk_kk-header-val-230905",
    "ktp_ktp-050921",
    "ktp_ktp-dataset",
    "npwp_npwp-0922",
    "npwp_npwp-bca-uat",
    "npwp_npwp-biru-06022023",
    "npwp_npwp-biru-putih-30-maret-2023",
    "npwp_npwp-putih-biru-2023-02-15-05-28-09",
    "others_npwp-all_text-20220805",
    "passport_passport-230523",
    "passport_passport-synthesis",
    "plate_plate-aws-wings",
    "receipt_struk-all_text-20220805",
    "seven_segments-1",
    "seven_segments-10",
    "seven_segments-11",
    "seven_segments-12",
    "seven_segments-13",
    "seven_segments-14",
    "seven_segments-2",
    "seven_segments-3",
    "seven_segments-4",
    "seven_segments-5",
    "seven_segments-6",
    "seven_segments-7",
    "seven_segments-8",
    "seven_segments-9",
    "sim_sim",
    "singapore-family-pass_sfp-synthesis",
    "singapore-nric_snric",
    "singapore-work-permit_swp-synthesis",
    "skpr_printed-all_text-20231023-6",
    "stnk_stnk-synthesis",
    "ymmi_yamaha-box"
]
configs['savedir'] = './models'
configs['imgH'] = 64
configs['imgW'] = 256

configs['alphabet'] = 'data/alphabet_en.txt'

f = open(configs.alphabet, 'r')
l = f.readline().rstrip()
f.close()
configs['n_class'] = len(l) + 3  # pad, unk, eos

configs['device'] = 'cpu'
configs['random_seed'] = 100
configs['batchsize'] = 64
configs['workers'] = 8

configs['n_epochs'] = 5
configs['lr'] = 0.5
configs['lr_milestones'] = [2, 5, 7]
configs['lr_gammas'] = [0.2, 0.1, 0.1]
configs['weight_decay'] = 0.

configs['aug_prob'] = 0.3
configs['continue_train'] = False
configs['continue_path'] = './models/pren.pth'
configs['displayInterval'] = 100

# ---- model
configs['net'] = edict()

configs.net['n_class'] = configs.n_class
configs.net['max_len'] = 280
configs.net['n_r'] = 5  # number of primitive representations
configs.net['d_model'] = 384
configs.net['dropout'] = 0.1
