from django.contrib import admin

from code_search_app import models


class CodeDocumentAdmin(admin.ModelAdmin):
    pass


class QueryLogAdmin(admin.ModelAdmin):
    pass


class CodeDocumentVisitLogAdmin(admin.ModelAdmin):
    pass


admin.site.register(models.CodeDocument, CodeDocumentAdmin)
admin.site.register(models.QueryLog, QueryLogAdmin)
admin.site.register(models.CodeDocumentVisitLog, CodeDocumentVisitLogAdmin)
