from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os,uuid
from .utils import _img_array_to_svg
import cv2
import numpy as np
from django.conf import settings


MEDIA_ROOT='media'
os.makedirs(MEDIA_ROOT,exist_ok=True)

def index(request):
    return render(request,'index.html')

@csrf_exempt
def generate_svg(request):
    uploaded = request.FILES["image"]

    k_min = int(request.POST.get("k_min", 3))
    k_max = int(request.POST.get("k_max", 15))
    cluster_scale = float(request.POST.get("cluster_scale", 0.5))
    min_area_ratio = float(request.POST.get("min_area_ratio", 0.0003))
    smooth = request.POST.get("smooth") == "true"

    # Decode image directly from memory
    image_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    svg_text = _img_array_to_svg(
        img,
        K_MIN=k_min,
        K_MAX=k_max,
        CLUSTER_SCALE=cluster_scale,
        MIN_AREA_RATIO=min_area_ratio,
        smooth=smooth
    )

    return HttpResponse(svg_text, content_type="image/svg+xml")
