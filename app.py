from PIL import Image
import torch
import gradio as gr
# Загрузка моделей с разными вариантами настроек
model1 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill")
model2 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
model3 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model4 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
def load_face2paint_model(size):
    global face2paint
    if size == 512:
        face2paint = torch.hub.load('bryandlee/animegan2-pytorch:main', 'face2paint',
                                    size=size, device="cpu", side_by_side=False)
    elif size == 1024:
        face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint",
                                    size=size, device="cpu", side_by_side=False)
load_face2paint_model(512)  # Загрузка модели с размером 512 по умолчанию
def inference(img, ver, size):
    if size != 512:
        load_face2paint_model(size)  # Загрузка модели с выбранным размером
    if ver == 'Стиль - 1':
        out = face2paint(model1, img)
    elif ver == 'Стиль - 2':
        out = face2paint(model2, img)
    elif ver == 'Стиль - 3':
        out = face2paint(model3, img)
    elif ver == 'Стиль - 4':
        out = face2paint(model4, img)
    return out
title = "Фото в Рисунок"
description = "Загрузите картинку любого размера до 1024px, выберите один из форматов генерации, выходной размер и нажмите кнопку Исполнить"
article = "<br><br><br><br><br>"
# Создание интерфейса Gradio с вариантами моделей, стилей и размеров и черным фоном
interface = gr.Interface(inference,
                         [gr.inputs.Image(type="pil"),
                          gr.inputs.Radio(['Стиль - 1', 'Стиль - 2', 'Стиль - 3', 'Стиль - 4'],
                                         type="value",
                                         default='Стиль - 1',
                                         label='Выбор стиля'),
                          gr.inputs.Radio([512, 1024],
                                         type="value",
                                         default=512,
                                         label='Выходной размер')],
                    gr.outputs.Image(type="pil"),
              title=title,
              description=description,
              article=article,
              allow_flagging=False,
              allow_screenshot=False).launch(debug=True, max_threads=True, share=True, inbrowser=True)
