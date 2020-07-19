from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings

from code_search_app import views, api_views

urlpatterns = [
    path(settings.ENV['ADMIN_PATH'], admin.site.urls),

    # API
    path('api/repositories', api_views.api_repositories_view),
    path('api/repositories/<str:repository_organization>/<str:repository_name>', api_views.api_repository_view),
    path('api/repositories/<str:repository_organization>/<str:repository_name>/search',
         api_views.api_repository_search_view),

    path('api/codeDocument/<str:repository_organization>/<str:repository_name>/<str:code_hash>',
         api_views.api_code_document_view),
    path('api/similarCodeDocuments/<str:repository_organization>/<str:repository_name>/<str:code_hash>',
         api_views.api_similar_code_documents_view),

    re_path(r'^.*/$', views.IndexView.as_view()),
    path('', views.IndexView.as_view()),
]
