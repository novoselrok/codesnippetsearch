import os
import json

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, '..', 'env.json')) as f:
    ENV = json.load(f)['WEB']

SECRET_KEY = ENV['SECRET_KEY']
DEBUG = ENV['DEBUG']
ALLOWED_HOSTS = ENV['ALLOWED_HOSTS']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'django.contrib.humanize',

    'corsheaders',

    'code_search_app.apps.CodeSearchAppConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'code_search_web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'client')
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'code_search_web.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': ENV['DB_NAME'],
        'USER': ENV['DB_USER'],
        'PASSWORD': ENV['DB_PASSWORD'],
        'HOST': ENV['DB_HOST'],
        'PORT': ENV['DB_PORT'],
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'client', 'dist', 'static'),
]

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
        'LOCATION': 'code_search_cache_table',
    }
}

if not DEBUG and len(ENV['SENTRY_DSN']) > 0:
    sentry_sdk.init(
        dsn=ENV['SENTRY_DSN'],
        integrations=[DjangoIntegration()]
    )

CORS_ORIGIN_REGEX_WHITELIST = [
    r'^https://\w+\.github\.com$',
    r'^https://\w+\.codesnippetsearch\.net$',
] + ENV['CORS_ORIGIN_REGEX_WHITELIST']
CORS_ALLOW_METHODS = (
    'GET',
    'OPTIONS',
    'POST'
)
