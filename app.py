"""Interactive three-grade AI Vision Lab."""
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
st.set_page_config(page_title='AI Vision Lab | M4–M6',page_icon='🔬',layout='wide')
st.title('🔬 AI Vision Lab')
st.caption('M4 Classification · M5 Custom Detection · M6 Model Comparison')
grade=st.sidebar.radio('Choose module',['M4 · Build a CNN','M5 · Custom Object Detection','M6 · Compare Experiments'])
if grade.startswith('M4'):
    st.header('M4 · Learn image classification')
    st.write('Train a small CNN **from scratch** on synthetic circles, squares and triangles.')
    path=Path('models/m4_shapes_cnn.pt')
    if not path.exists():st.warning('First train the model: `python scripts/train_m4.py --epochs 8`')
    else:
        import torch
        from scripts.train_m4 import TinyCNN
        @st.cache_resource
        def load(p):
            ck=torch.load(p,map_location='cpu',weights_only=True)
            m=TinyCNN();m.load_state_dict(ck['model_state']);m.eval()
            return m,ck['classes']
        model,classes=load(str(path))
        mode=st.radio('Image source',['Sample shape','Upload image'],horizontal=True)
        if mode=='Sample shape':
            from visionlab.shapes import make_image,CLASSES
            label=st.selectbox('Shape',CLASSES);seed=st.slider('Variation seed',0,999,11)
            im=make_image(CLASSES.index(label),np.random.default_rng(seed))
        else:
            file=st.file_uploader('Upload image',type=['jpg','jpeg','png'])
            im=Image.open(file).convert('RGB') if file else None
        if im is not None:
            left,right=st.columns(2);left.image(im,caption='Input image',width=260)
            x=np.asarray(im.resize((64,64)).convert('RGB'),dtype=np.float32).transpose(2,0,1).copy()/255
            with torch.no_grad():probs=torch.softmax(model(torch.from_numpy(x).unsqueeze(0)),1)[0].numpy()
            right.subheader('Prediction: '+classes[int(probs.argmax())])
            for name,p in zip(classes,probs):right.progress(float(p),text=f'{name}: {p:.1%}')
        hist=Path('models/m4_training_history.csv')
        if hist.exists():
            import pandas as pd
            st.line_chart(pd.read_csv(hist).set_index('epoch')[['train_accuracy','val_accuracy']])
        st.info('Discuss how novel shapes differ from the training examples.')
elif grade.startswith('M5'):
    st.header('M5 · Custom object detection')
    st.write('Fine-tune YOLO with your own annotated classroom-object dataset.')
    weights=Path('runs/m5/custom_objects/weights/best.pt')
    st.code('python scripts/train_m5.py --data data/custom_objects/data.yaml --epochs 30 --device cpu')
    if not weights.exists():st.warning('No custom detector yet. Add annotated images and train first.')
    else:
        uploaded=st.file_uploader('Image to detect',type=['png','jpg','jpeg'])
        conf=st.slider('Confidence threshold',.05,.95,.25)
        if uploaded:
            from ultralytics import YOLO
            @st.cache_resource
            def get_yolo(p):return YOLO(p)
            result=get_yolo(str(weights)).predict(source=Image.open(uploaded).convert('RGB'),conf=conf,verbose=False)[0]
            st.image(result.plot()[:,:,::-1],caption='Custom detections',use_container_width=True)
            st.write('Objects:',len(result.boxes))
    st.markdown('**Student prompt:** How does annotation quality affect the model?')
else:
    st.header('M6 · Compare experiments')
    st.write('Evaluate two different checkpoints on the same labeled hold-out images.')
    st.code('python scripts/evaluate_m6.py --data data/custom_objects/data.yaml --models path/to/first.pt path/to/second.pt --split test')
    result=Path('runs/m6/model_comparison.csv')
    if result.exists():
        import pandas as pd
        df=pd.read_csv(result);st.dataframe(df,use_container_width=True)
        st.bar_chart(df.set_index('checkpoint')[['precision','recall','mAP50','mAP50_95']])
    else:st.warning('No M6 model comparison exists yet. Train and evaluate two models.')
    st.markdown('**Student prompt:** Does more training always improve test-set performance?')
