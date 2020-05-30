from django.contrib import admin
from django.db.models import Count
from django.db.models.functions import TruncDay

from code_search_app import models


class CodeDocumentAdmin(admin.ModelAdmin):
    pass


class CodeDocumentQueryRatingAdmin(admin.ModelAdmin):
    raw_id_fields = ('code_document',)


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


class CodeDocumentVisitLogAdmin(admin.ModelAdmin):
    raw_id_fields = ('code_document',)

    def changelist_view(self, request, extra_context=None):
        # Aggregate code document visits
        visits_per_day = models.CodeDocumentVisitLog.objects.annotate(date=TruncDay('created_at')) \
            .values('date') \
            .annotate(n_visits=Count('id')) \
            .order_by('-date') \

        extra_context = extra_context or {'visits_per_day': list(visits_per_day)[:5]}
        # Call the superclass changelist_view to render the page
        return super().changelist_view(request, extra_context=extra_context)


admin.site.register(models.CodeDocument, CodeDocumentAdmin)
admin.site.register(models.QueryLog, QueryLogAdmin)
admin.site.register(models.CodeDocumentVisitLog, CodeDocumentVisitLogAdmin)
admin.site.register(models.CodeDocumentQueryRating, CodeDocumentQueryRatingAdmin)
