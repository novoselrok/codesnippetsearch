from django.contrib import admin
from django.db.models import Count
from django.db.models.functions import TruncDay

from code_search_app import models


class CodeDocumentAdmin(admin.ModelAdmin):
    pass


class CodeRepositoryAdmin(admin.ModelAdmin):
    pass


class QueryLogAdmin(admin.ModelAdmin):

    def changelist_view(self, request, extra_context=None):
        # Aggregate query logs
        logs_per_day = models.QueryLog.objects.annotate(date=TruncDay('created_at')) \
            .values('date') \
            .annotate(n_logs=Count('id')) \
            .order_by('-date') \

        extra_context = extra_context or {'logs_per_day': list(logs_per_day)[:5]}
        # Call the superclass changelist_view to render the page
        return super().changelist_view(request, extra_context=extra_context)


admin.site.register(models.CodeDocument, CodeDocumentAdmin)
admin.site.register(models.QueryLog, QueryLogAdmin)
admin.site.register(models.CodeRepository, CodeRepositoryAdmin)
