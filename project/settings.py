
import os

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-&psk#na5l=p3q8_a+-$4w1f^lt3lx1c@d*p4x$ymm_rn7pwb87')
DEBUG = os.environ.get('DJANGO_DEBUG', '') != 'False'
ALLOWED_HOSTS=[]
INSTALLED_APPS=['django.contrib.staticfiles','converter']
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
ROOT_URLCONF='project.urls'
TEMPLATES=[{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':['converter/templates'],'APP_DIRS':True}]
WSGI_APPLICATION='project.wsgi.application'
STATIC_URL='/static/'
STATICFILES_DIRS=['converter/static']
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}
