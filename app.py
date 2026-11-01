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
    st.write('Explore a synthetic shape dataset, train a small CNN **from scratch**, and investigate its predictions.')
    from visionlab.shapes import make_image, CLASSES
    import torch
    import pandas as pd
    preview, guide = st.columns([2, 1])
    with guide:
        st.subheader('Learning objectives')
        st.markdown('1. Recognize image labels and classes.\n2. Distinguish training, validation and test data.\n3. Interpret confidence scores and errors.')
        st.caption('Synthetic shapes are used for teaching, not real-world recognition.')
    with preview:
        st.subheader('Explore your data')
        label = st.selectbox('Choose the true shape', CLASSES)
        seed = st.slider('Change sample appearance', 0, 999, 11)
        im = make_image(CLASSES.index(label), np.random.default_rng(seed))
        st.image(im.resize((256, 256), resample=Image.Resampling.NEAREST),
                 caption=f'Synthetic sample: {label}', width=256)
    checkpoint = Path('models/m4_shapes_cnn.pt')
    st.subheader('Train the model')
    if not checkpoint.is_file():
        st.info('The model checkpoint is not on this computer yet. You can explore the training images above before training.')
        st.code('python scripts/train_m4.py --epochs 8', language='bash')
        st.caption('Run this in Terminal from the repository root, then refresh this page. The training script creates the dataset and the checkpoint.')
    else:
        from scripts.train_m4 import TinyCNN
        @st.cache_resource
        def load_m4(path, last_modified):
            ck = torch.load(path, map_location='cpu', weights_only=True)
            model = TinyCNN()
            model.load_state_dict(ck['model_state'])
            model.eval()
            return model, ck['classes']
        model, classes = load_m4(str(checkpoint), checkpoint.stat().st_mtime)
        st.success('Trained CNN checkpoint found.')
        source = st.radio('Test input', ['Generated example', 'Upload image'], horizontal=True)
        uploaded = st.file_uploader('Upload a shape image', type=['jpg', 'jpeg', 'png']) if source == 'Upload image' else None
        test_image = Image.open(uploaded).convert('RGB') if uploaded else (im if source == 'Generated example' else None)
        if test_image is not None:
            data = np.asarray(test_image.resize((64, 64)).convert('RGB'), dtype=np.float32).transpose(2, 0, 1).copy() / 255
            with torch.no_grad():
                probabilities = torch.softmax(model(torch.from_numpy(data).unsqueeze(0)), 1)[0].numpy()
            prediction = classes[int(probabilities.argmax())]
            st.subheader(f'Prediction: {prediction}')
            for name, probability in zip(classes, probabilities):
                st.progress(float(probability), text=f'{name}: {probability:.1%}')
            st.caption('The percentages are model confidence scores, not guarantees of correctness.')
    history = Path('models/m4_training_history.csv')
    metrics = Path('models/m4_test_metrics.json')
    if history.is_file():
        st.subheader('Training progress')
        frame = pd.read_csv(history)
        tab1, tab2 = st.tabs(['Accuracy', 'Loss'])
        with tab1:
            st.line_chart(frame.set_index('epoch')[['train_accuracy', 'val_accuracy']])
        with tab2:
            st.line_chart(frame.set_index('epoch')[['train_loss', 'val_loss']])
    if metrics.is_file():
        import json
        report = json.loads(metrics.read_text())
        st.metric('Held-out test accuracy', f"{report['test_accuracy']:.1%}")
        matrix = report.get('confusion_matrix_rows_true_cols_pred')
        if matrix is not None:
            st.subheader('Confusion matrix')
            st.caption('Rows: true class · Columns: predicted class')
            st.dataframe(pd.DataFrame(matrix, index=report['classes'], columns=report['classes']))
    with st.expander('Teacher questions'):
        st.markdown('- How is a **classification** prediction different from an **object detection** box?\n- Why must we keep test images separate from training images?\n- What happens if the training images are much simpler than real photos?')
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
