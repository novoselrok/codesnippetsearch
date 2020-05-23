from django.contrib import admin
from django.urls import path
from django.conf import settings

from code_search_app import views

urlpatterns = [
    path(settings.ENV['ADMIN_PATH'], admin.site.urls),

    path('', views.index_view, name='index'),
    path('search', views.search_view, name='search'),
    path('code/<str:code_hash>/visit', views.code_document_visit_view, name='code_document_visit'),
    path('code/<str:code_hash>', views.code_document_view, name='code_document'),
]
