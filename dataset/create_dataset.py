# This script creates datasets. We'll link every image in the target domain for ACDC experiemnts, and select only 25
# by using the dataloader. This was necessary to run ablation studies on seeds.

import os
from argparse import ArgumentParser as AP

# Viper sequence codes
day_codes = ['001', '002', '003', '004', '005',
             '006', '044', '045', '046', '047',
             '048', '049', '050', '051', '065',
             '066', '067', '068', '069', ]
sunset_codes = ['007', '014', '015', '016', '017', '018',
                '019', '020', '021', '022', '023', '024',
                '025', '026', '027', '028', '029']
night_codes = ['008', '009', '010', '011', '012', '013',
               '052', '053', '054', '055', '056', '057',
               '058', '070', '071', '072', '073', '074',
               '075', '076', '077', ]
rain_codes = ['030', '031', '059', '060', '061', '062', '063', '064']
snow_codes = ['032', '033', '034', '035', '036', '037',
              '038', '039', '040', '041', '042', '043']

def get_base_acdc(root, condition, subset):
    base = os.path.join(root, 'rgb_anon', condition)
    base_source = os.path.join(base, '{}_ref'.format(subset))
    base_target = os.path.join(base, '{}'.format(subset))
    return base, base_source, base_target

def create_ds(root, viper_root, name, subset, condition, anchor, type='acdc'):
    os.makedirs(name, exist_ok=True)
    os.makedirs('{}/{}S'.format(name, subset), exist_ok=True)
    os.makedirs('{}/{}T'.format(name, subset), exist_ok=True)
    os.makedirs('{}/{}A'.format(name, subset), exist_ok=True)
    if type == 'acdc':
        base, base_source, base_target = get_base_acdc(root, condition, subset)
        link(base_source, '{}/{}S'.format(name, subset))
        link(base_target, '{}/{}T'.format(name, subset))
    elif type == 'dz':
        with open('list_dz_twilight.txt') as file:
            dz_gt_set = file.read().splitlines()
        base_source = os.path.join(root, 'rgb_anon', 'train', 'day')
        link(base_source, '{}/{}S'.format(name, subset))
        base_target = os.path.join(root, 'rgb_anon', 'train', 'twilight', 'GOPR0348')
        files_target = [os.path.join(base_target, x) for x in dz_gt_set]
        for f in files_target:
            os.symlink(f,  '{}/{}T/{}'.format(name, subset, os.path.basename(f)))

    viper_root = os.path.join(viper_root, '{}/img'.format(subset))
    print(anchor)
    if anchor == 'day':
        seqs = day_codes
    elif anchor == 'night':
        seqs = night_codes
    elif anchor == 'rain':
        seqs = rain_codes
    elif anchor == 'sunset':
        seqs = sunset_codes
    elif anchor == 'snow':
        seqs = snow_codes
    else:
        raise('Wrong anchor!')

    # create synthetic anchor
    for x in os.listdir(viper_root):
        dir_path = os.path.join(viper_root, x)
        if x not in seqs:
            print("{} skipped".format(x))
            continue
        for y in os.listdir(dir_path):
            os.symlink(os.path.join(dir_path, y), os.path.join(name, '{}A'.format(subset), y))


def link(directory, output_dir):
    for x in os.listdir(directory):
        dir_path = os.path.join(directory, x)
        if os.path.isdir(dir_path):
            for y in os.listdir(dir_path):
                os.symlink(os.path.join(dir_path, y), os.path.join(output_dir, y))

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--root_acdc', help='ACDC root', required=True, type=str)
    ap.add_argument('--root_dz', help='DZ root', required=True, type=str)
    ap.add_argument('--root_viper', help='Viper root', required=True, type=str)
    opt = ap.parse_args()
    root_acdc = opt.root_acdc
    root_viper = opt.root_viper

    create_ds(opt.root_acdc, opt.root_viper, 'day2night', 'train', 'night', 'night')
    create_ds(opt.root_acdc, opt.root_viper, 'clear2fog', 'train', 'fog', 'day')
    create_ds(opt.root_dz, opt.root_viper, 'day2twilight', 'train', None, 'night', type='dz')

