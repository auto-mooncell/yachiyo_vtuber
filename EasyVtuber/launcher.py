import os
import subprocess
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk
import json
import sys

cache_simplify_map = {
    'Off':0,
    'Low':1,
    'Medium':2,
    'High':3,
    'Higher':4,
    'Highest':6,
    'Gaming':8
}

cache_simplify_quality_map = {
    'Off':100,
    'Low':99,
    'Medium':95,
    'High':90,
    'Higher':85,
    'Highest':80,
    'Gaming':75
}

default_arg = {
    'character': 'lambda_00',
    'input': 2,
    'output': 2,
    'ifm': None,
    'osf': '127.0.0.1:11573',
    'is_extend_movement': False,
    'is_alpha_split': False,
    'is_bongo': False,
    'is_eyebrow': False,
    'cache_simplify': 'High',
    'ram_cache_size': '2gb',
    'vram_cache_size': '2gb',
    'model_select':'seperable_half',
    'interpolation':'x2_half',
    'frame_rate_limit':'30',
    'sr':'Off',
    'device_id':'0',
    'use_tensorrt':True
}

try:
    f = open('launcher.json')
    args = json.load(f)
    default_arg.update(args)
    f.close()
except:
    pass
finally:
    args = default_arg

p = None
dirPath = 'data/images'
characterList = []
for item in sorted(os.listdir(dirPath), key=lambda x: -os.path.getmtime(os.path.join(dirPath, x))):
    if '.png' == item[-4:]:
        characterList.append(item[:-4])

root = tk.Tk()
root.resizable(False, False)
root.title('EasyVtuber Launcher')

launcher = ttk.Frame(root)
launcher.pack(fill='x', expand=True)


def launch():
    global p
    global launch_btn
    args = {
        'character': character.get(),
        'input': input.get(),
        'output': output.get(),
        'ifm': ifm.get(),
        'osf': osf.get(),
        'is_extend_movement': is_extend_movement.get(),
        'is_alpha_split': is_alpha_split.get(),
        'is_bongo': is_bongo.get(),
        'is_eyebrow': is_eyebrow.get(),
        'cache_simplify': cache_simplify.get(),
        'ram_cache_size': ram_cache_size.get(),
        'vram_cache_size': vram_cache_size.get(),
        'model_select': model_select.get(),
        'interpolation':interpolation.get(),
        'frame_rate_limit':frame_rate_limit.get(),
        'sr':sr.get(),
        'device_id':device_id.get(),
        'use_tensorrt':use_tensorrt.get()
    }

    if args['input'] == 0:
        if len(args['ifm']) == 0:
            tkinter.messagebox.showinfo('EasyVtuber Launcher', 'Please Input iFacialMocap IP:Port')
            return
    if args['input'] == 4:
        if len(args['osf']) == 0:
            tkinter.messagebox.showinfo('EasyVtuber Launcher', 'Please Input OpenSeeFace IP:Port')
            return

    f = open('launcher.json', mode='w')
    json.dump(args, f)
    f.close()
    if p is not None:
        subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        p = None
        launch_btn.config(text="Save & Launch")
    else:
        run_args = [sys.executable, 'main.py']
        if len(args['character']):
            run_args.append('--character')
            run_args.append(args['character'])

        if args['input'] == 0:
            if len(args['ifm']):
                run_args.append('--ifm')
                run_args.append(args['ifm'])
        elif args['input'] == 1:
            run_args.append('--input')
            run_args.append('cam')
        elif args['input'] == 2:
            run_args.append('--debug_input')
        elif args['input'] == 3:
            run_args.append('--mouse_input')
            run_args.append('0,0,' + str(root.winfo_screenwidth()) + ',' + str(root.winfo_screenheight()))
        elif args['input'] == 4:
            if len(args['osf']):
                run_args.append('--osf')
                run_args.append(args['osf'])

        if args['output'] == 0:
            run_args.append('--output_webcam')
            run_args.append('unitycapture')
        elif args['output'] == 1:
            run_args.append('--output_webcam')
            run_args.append('obs')
        elif args['output'] == 3:
            run_args.append('--output_webcam')
            run_args.append('spout')
        elif args['output'] == 2:
            run_args.append('--debug')
        if args['is_alpha_split']:
            run_args.append('--alpha_split')
        if args['is_extend_movement']:
            run_args.append('--extend_movement')
            run_args.append('1')
        if args['is_bongo']:
            run_args.append('--bongo')
        if args['is_eyebrow']:
            run_args.append('--eyebrow')
        if args['cache_simplify'] is not None:
            run_args.append('--simplify')
            run_args.append(str(cache_simplify_map[args['cache_simplify']]))
            run_args.append('--cacher_quality')
            run_args.append(str(cache_simplify_quality_map[args['cache_simplify']]))
            if args['cache_simplify'] != 'Off':
                run_args.append('--use_cacher')
        if args['ram_cache_size'] is not None:
            run_args.append('--cache')
            run_args.append(args['ram_cache_size'])
            run_args.append('--gpu_cache')
            run_args.append(args['vram_cache_size'])

        if args['interpolation'] is not None:
            if not 'off' == args['interpolation']:
                run_args.append('--use_interpolation')
            if 'half' in args['interpolation']:
                run_args.append('--interpolation_half')
                
            if 'x2' in  args['interpolation']:
                run_args.append('--interpolation_scale')
                run_args.append('2')
            elif 'x3' in  args['interpolation']:
                run_args.append('--interpolation_scale')
                run_args.append('3')
            elif 'x4' in  args['interpolation']:
                run_args.append('--interpolation_scale')
                run_args.append('4')

        if args['model_select'] is not None:
            if 'seperable' in args['model_select']:
                run_args.append('--model_seperable')
            if 'half' in  args['model_select']:
                run_args.append('--model_half')

        if args['frame_rate_limit'] is not None:
            run_args.append('--frame_rate_limit')
            run_args.append(args['frame_rate_limit'])
            
        if args['sr'] is not None and args['sr'] != 'Off':
            if 'anime4k' in args['sr']:
                run_args.append('--anime4k')
            else:
                run_args.append('--use_sr')
                if 'x4' in args['sr']:
                    run_args.append('--sr_x4')
                if 'half' in args['sr']:
                    run_args.append('--sr_half')

        if args['device_id'] is not None:
            run_args.append('--device_id')
            run_args.append(args['device_id'])

        if args['use_tensorrt'] is not None and args['use_tensorrt']:
            run_args.append('--use_tensorrt')
            run_args.append('--model_cache')
            run_args.append('--model_vram_cache')

        run_args.append('--output_size')
        run_args.append('512x512')
        print('Launched: ' + ' '.join(run_args))
        p = subprocess.Popen(run_args)
        launch_btn.config(text='Stop')


launch_btn = ttk.Button(launcher, text="Save & Launch", command=launch)
launch_btn.pack(side='bottom', fill='x', expand=True, pady=10, padx=10)

frameL = ttk.Frame(launcher)
frameL.pack(padx=10, pady=10, fill='both', side='left', expand=True)
frameR = ttk.Frame(launcher)
frameR.pack(padx=10, pady=10, fill='both', side='left', expand=True)

character = tk.StringVar(value=args['character'])
ttk.Label(frameL, text="Character").pack(fill='x', expand=True)

# ttk.Entry(frameL, textvariable=character).pack(fill='x', expand=True)
char_combo = ttk.Combobox(frameL, textvariable=character, value=characterList)
char_combo.pack(fill='x', expand=True)



def inputChange():
    i=input.get()
    if i==0:
        ifmLbl.pack(fill='x', expand=True)
        ifmEnt.pack(fill='x', expand=True)
        osfLbl.pack_forget()
        osfEnt.pack_forget()
    elif i==4:
        ifmLbl.pack_forget()
        ifmEnt.pack_forget()
        osfLbl.pack(fill='x', expand=True)
        osfEnt.pack(fill='x', expand=True)
    else:
        ifmLbl.pack_forget()
        ifmEnt.pack_forget()
        osfLbl.pack_forget()
        osfEnt.pack_forget()

input = tk.IntVar(value=args['input'])
ttk.Label(frameL, text="Face Data Source").pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='iFacialMocap', value=0, variable=input, command=inputChange).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='OpenSeeFace', value=4, variable=input, command=inputChange).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Webcam(opencv)', value=1, variable=input, command=inputChange).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Mouse Input', value=3, variable=input, command=inputChange).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Initial Debug Input', value=2, variable=input, command=inputChange).pack(fill='x',
                                                                                                       expand=True)
frameLTxt = ttk.Frame(frameL)
frameLTxt.pack(fill='x', expand=True)
ifmLbl = ttk.Label(frameLTxt, text="iFacialMocap IP:Port")
ifmLbl.pack(fill='x', expand=True)

ifm = tk.StringVar(value=args['ifm'])
ifmEnt = ttk.Entry(frameLTxt, textvariable=ifm, state=False)
ifmEnt.pack(fill='x', expand=True)

osfLbl = ttk.Label(frameLTxt, text="OpenSeeFace IP:Port")
osfLbl.pack(fill='x', expand=True)
osf = tk.StringVar(value=args['osf'])
osfEnt = ttk.Entry(frameLTxt, textvariable=osf, state=False)
osfEnt.pack(fill='x', expand=True)
inputChange()

ttk.Label(frameR, text="GPU Device ID").pack(fill='x', expand=True)
device_id = tk.StringVar(value=args['device_id'])
device_id_combo = ttk.Combobox(frameR, textvariable=device_id, value=['0', '1', '2', '3', '4'], state='readonly').pack(fill='x', expand=True)

use_tensorrt = tk.BooleanVar(value=args['use_tensorrt'])
ttk.Checkbutton(frameR, text='Use TensorRT', variable=use_tensorrt).pack(fill='x', expand=True)

ttk.Label(frameR, text="Model Select").pack(fill='x', expand=True)
model_select = tk.StringVar(value=args['model_select'])
model_select_combo = ttk.Combobox(frameR, textvariable=model_select, value=['seperable_half', 'seperable_full','standard_half','standard_full'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="VRAM Cache Size (TensorRT only)").pack(fill='x', expand=True)
vram_cache_size = tk.StringVar(value=args['vram_cache_size'])
vram_cache_size_combo = ttk.Combobox(frameR, textvariable=vram_cache_size, value=['0b', '128mb', '256mb','512mb', '1gb', '2gb', '4gb'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="Facial Input Simplify & Cache Quality").pack(fill='x', expand=True)
cache_simplify = tk.StringVar(value=args['cache_simplify'])
cache_simplify_combo = ttk.Combobox(frameR, textvariable=cache_simplify, value=['Off', 'Low','Medium','High', 'Higher', 'Highest', 'Gaming'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="RAM Cache Size").pack(fill='x', expand=True)
ram_cache_size = tk.StringVar(value=args['ram_cache_size'])
ram_cache_size_combo = ttk.Combobox(frameR, textvariable=ram_cache_size, value=['0b', '1gb', '2gb', '4gb', '8gb', '16gb'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="Frame Interpolation").pack(fill='x', expand=True)
interpolation = tk.StringVar(value=args['interpolation'])
interpolation_combo = ttk.Combobox(frameR, textvariable=interpolation, value=['off', 'x2_half', 'x3_half','x4_half', 'x2_full', 'x3_full', 'x4_full'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="Frame Rate Limitation").pack(fill='x', expand=True)
frame_rate_limit = tk.StringVar(value=args['frame_rate_limit'])
frame_rate_limit_combo = ttk.Combobox(frameR, textvariable=frame_rate_limit, value=['20', '25', '30','40', '50', '60'], state='readonly').pack(fill='x', expand=True)

ttk.Label(frameR, text="Super Resolution").pack(fill='x', expand=True)
sr = tk.StringVar(value=args['sr'])
sr_combo = ttk.Combobox(frameR, textvariable=sr, value=['Off', 'anime4k_x2', 'waifu2x_x2_half', 'real-esrgan_x4_half', 'waifu2x_x2_full', 'real-esrgan_x4_full'], state='readonly').pack(fill='x', expand=True)



ttk.Label(frameL, text="Extra Options").pack(fill='x', expand=True)
is_eyebrow = tk.BooleanVar(value=args['is_eyebrow'])
ttk.Checkbutton(frameL, text='Eyebrow (iFM Only)', variable=is_eyebrow).pack(fill='x', expand=True)

is_extend_movement = tk.BooleanVar(value=args['is_extend_movement'])
ttk.Checkbutton(frameL, text='Extend Movement', variable=is_extend_movement).pack(fill='x', expand=True)


is_alpha_split = tk.BooleanVar(value=args['is_alpha_split'])
ttk.Checkbutton(frameL, text='Alpha Split', variable=is_alpha_split).pack(fill='x', expand=True)

is_bongo = tk.BooleanVar(value=args['is_bongo'])
ttk.Checkbutton(frameL, text='Bongocat Mode', variable=is_bongo).pack(fill='x', expand=True)

output = tk.IntVar(value=args['output'])
ttk.Label(frameL, text="Output").pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Unity Capture', value=0, variable=output).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='OBS Virtual Camera', value=1, variable=output).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Spout2', value=3, variable=output).pack(fill='x', expand=True)
ttk.Radiobutton(frameL, text='Initial Debug Output', value=2, variable=output).pack(fill='x', expand=True)


def closeWindow():
    if p is not None:
        subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    root.destroy()


def handle_focus(event):
    characterList = []
    if event.widget == root:
        for item in sorted(os.listdir(dirPath), key=lambda x: -os.path.getmtime(os.path.join(dirPath, x))):
            if '.png' == item[-4:]:
                characterList.append(item[:-4])
        char_combo.config(value=characterList)


root.bind("<FocusIn>", handle_focus)
root.protocol('WM_DELETE_WINDOW', closeWindow)
root.mainloop()
