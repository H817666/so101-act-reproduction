"""Independent Tk preview process and lossless terminal/window command routing."""
import multiprocessing as mp
import os
import queue
import select
import time
from collections import deque
from pathlib import Path


def normalize_key(key):
    key = key.lower()
    return {'return': '\n', 'kp_enter': '\n', '\r': '\n',
            'space': ' ', 'escape': 'q', '\x03': 'q'}.get(key, key)


class CommandInput:
    def __init__(self, fd):
        self.fd = fd
        self.pending = deque()

    def read(self, window_keys=()):
        self.pending.extend(normalize_key(k) for k in window_keys)
        if select.select([self.fd], [], [], 0)[0]:
            # TextIOWrapper.read(1) can prefetch several keys, leaving select()
            # blind to the remaining characters in Python's private buffer.
            data = os.read(self.fd, 4096)
            if not data:
                return 'q'
            self.pending.extend(normalize_key(chr(k)) for k in data)
        if any(k in ('q', ' ') for k in self.pending):
            self.pending.clear()
            return 'q'
        return self.pending.popleft() if self.pending else ''


def drain(channel):
    items = []
    while True:
        try:
            items.append(channel.get_nowait())
        except queue.Empty:
            return items


def replace_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            return
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


class RecorderWindow:
    """All Tk calls and image scaling live here, never in the motor loop."""
    def __init__(self, commands, states, frames):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.font import families
        from PIL import Image, ImageTk, ImageDraw, ImageFont
        self.Image, self.ImageTk, self.ImageDraw, self.ImageFont = Image, ImageTk, ImageDraw, ImageFont
        self.commands, self.states, self.frames = commands, states, frames
        self.root = tk.Tk()
        font = 'Noto Sans CJK SC' if 'Noto Sans CJK SC' in families(self.root) else 'Sans'
        # This virtualenv's Tk uses legacy X fonts and can resolve Noto to
        # "fixed", losing Chinese glyphs. Pillow can load the installed TTC.
        self.cjk_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        self.render_cjk = self.cjk_path.is_file()
        self.font_cache = {}
        self.root.title('SO101 ACT 采集 · 同步状态 / 20Hz')
        width = min(1680, max(900, self.root.winfo_screenwidth()-160))
        height = min(1320, max(720, self.root.winfo_screenheight()-120))
        self.root.geometry(f'{width}x{height}')
        self.root.minsize(800, 650)
        style = ttk.Style(self.root)
        style.configure('SO101.TButton', font=(font, 14), padding=(8, 8))
        self.status = tk.StringVar(value='正在连接采集程序…')
        self.notice = tk.StringVar(value='')
        self.path = tk.StringVar(value='')
        self.status_label = ttk.Label(self.root, textvariable=self.status, font=(font, 18, 'bold'),
                                     wraplength=width-40)
        self.status_label.pack(fill='x', padx=16, pady=8)
        self.canvas = tk.Canvas(self.root, bg='#101820', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=12)
        self.image_item = self.canvas.create_image(0, 0, anchor='center')
        self.notice_label = tk.Label(self.root, textvariable=self.notice, font=(font, 15),
                                     justify='left', anchor='w', wraplength=width-40, bg='#eef4f8')
        self.notice_label.pack(fill='x', padx=12, pady=8)
        bar = ttk.Frame(self.root)
        bar.pack(fill='x', padx=8, pady=4)
        self.buttons = {}
        self.button_titles = {}
        for title, key in [('原位保持 ↵','\n'), ('跟随 R','r'), ('录制 C','c'),
                           ('保存 S','s'), ('丢弃 X','x'), ('暂停 P','p'), ('结束 Q','q')]:
            button = ttk.Button(bar, text=title, style='SO101.TButton',
                                command=lambda k=key: self.submit(k))
            button.pack(side='left', fill='x', expand=True, padx=3)
            self.buttons[key] = button
            self.button_titles[key] = title
        self.path_label = ttk.Label(self.root, textvariable=self.path, font=(font, 11), wraplength=width-40)
        self.path_label.pack(fill='x', padx=12, pady=4)
        if self.render_cjk:
            for label in (self.status_label,self.notice_label,self.path_label):
                label.configure(textvariable='', text='')
        self.root.bind('<Configure>', self.on_resize)
        self.root.bind_all('<KeyPress>', self.on_key)
        self.root.protocol('WM_DELETE_WINDOW', lambda: self.submit('q'))
        self.last_image = None
        self.last_size = None
        self.last_status_time = time.monotonic()
        self.snapshot = {}
        self.render_labels()
        self.root.after(20, self.tick)

    def text_image(self, widget, text, size, color, background, width):
        if not self.render_cjk:
            return
        signature = (text,size,color,background,width)
        if getattr(widget,'_text_signature',None) == signature:
            return
        if size not in self.font_cache:
            self.font_cache[size] = self.ImageFont.truetype(str(self.cjk_path),size,index=2)
        font = self.font_cache[size]
        lines, line = [], ''
        for char in text:
            if char == '\n' or (line and font.getlength(line+char)>width-8):
                lines.append(line)
                line = ''
            if char != '\n':
                line += char
        lines.append(line)
        ascent, descent = font.getmetrics()
        line_height = ascent+descent+3
        image = self.Image.new('RGB',(width,line_height*len(lines)+6),background)
        draw = self.ImageDraw.Draw(image)
        for i,line in enumerate(lines):
            draw.text((4,3+i*line_height),line,font=font,fill=color,anchor='lt')
        widget._text_photo = self.ImageTk.PhotoImage(image)
        widget.configure(image=widget._text_photo)
        widget._text_signature = signature

    def render_labels(self):
        if not self.render_cjk:
            return
        width = max(760,self.root.winfo_width()-40)
        bg = '#d9d9d9'
        self.text_image(self.status_label,self.status.get(),24,'#152c3d',bg,width)
        self.text_image(self.notice_label,self.notice.get(),21,
                        '#a32216' if self.snapshot.get('warning') else '#153a50','#eef4f8',width)
        self.text_image(self.path_label,self.path.get(),16,'#334155',bg,width)
        for key,button in self.buttons.items():
            self.text_image(button,self.button_titles[key],18,
                            '#888888' if button.instate(['disabled']) else '#152c3d',bg,122)

    def on_resize(self, event):
        if event.widget is self.root:
            for label in (self.status_label, self.notice_label, self.path_label):
                label.configure(wraplength=max(200, event.width-40))
            self.render_labels()

    def submit(self, key):
        key = normalize_key(key)
        if key not in ('\n', 'r', 'c', 's', 'x', 'p', 'q', ' ', 'g'):
            return
        try:
            self.commands.put_nowait(key)
        except queue.Full:
            self.notice.set('操作队列已满，请等待程序响应。')

    def on_key(self, event):
        key = normalize_key(event.keysym)
        if key in ('\n', 'r', 'c', 's', 'x', 'p', 'q', ' ', 'g'):
            self.submit(key)
            return 'break'

    def tick(self):
        states = drain(self.states)
        if states:
            self.snapshot = states[-1]
            self.last_status_time = time.monotonic()
            if self.snapshot.get('close'):
                self.root.destroy()
                return
            self.status.set(self.snapshot['summary'])
            self.notice.set(self.snapshot['notice'])
            self.path.set(self.snapshot['path'])
            self.notice_label.configure(fg='#a32216' if self.snapshot.get('warning') else '#153a50')
            allowed = self.snapshot.get('allowed', [])
            for key, button in self.buttons.items():
                button.configure(state='normal' if key in allowed else 'disabled')
        elif time.monotonic()-self.last_status_time > 2:
            self.status.set('等待控制程序响应… 请勿重复操作；Q 可请求停止')
        self.render_labels()
        frames = drain(self.frames)
        size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        if frames:
            self.last_image = frames[-1]
        if self.last_image is not None and (frames or size != self.last_size):
            scale = min(size[0]/640, size[1]/480)
            image = self.Image.fromarray(self.last_image).resize(
                (max(1, int(640*scale)), max(1, int(480*scale))))
            self.photo = self.ImageTk.PhotoImage(image)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            self.canvas.coords(self.image_item, size[0]//2, size[1]//2)
            self.last_size = size
        self.root.after(33, self.tick)


def run_window(commands, states, frames, ready):
    try:
        window = RecorderWindow(commands, states, frames)
        ready.put(None)
        window.root.mainloop()
    except BaseException as exc:
        ready.put(str(exc))


class PreviewProcess:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.commands = ctx.Queue(maxsize=128)
        self.states = ctx.Queue(maxsize=2)
        self.frames = ctx.Queue(maxsize=1)
        self.ready = ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=run_window,
                                   args=(self.commands, self.states, self.frames, self.ready), daemon=True)
        self.process.start()
        try:
            error = self.ready.get(timeout=15)
            if error:
                raise RuntimeError(f'预览窗口启动失败：{error}')
        except BaseException:
            self.close()
            raise

    def keys(self):
        if not self.process.is_alive():
            raise RuntimeError('预览窗口已意外关闭，控制已停止；未保存片段将保留备份')
        return drain(self.commands)

    def publish(self, snapshot, image=None):
        replace_latest(self.states, snapshot)
        if image is not None:
            replace_latest(self.frames, image)

    def close(self):
        if self.process.is_alive():
            replace_latest(self.states, {'close': True})
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        for channel in (self.commands, self.states, self.frames, self.ready):
            channel.cancel_join_thread()
            channel.close()
