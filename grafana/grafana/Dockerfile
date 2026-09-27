FROM grafana/grafana:10.4.2

# Copy provisioning configs (datasources + dashboard provider)
COPY provisioning /etc/grafana/provisioning

# Copy pre-built dashboards
COPY dashboards /var/lib/grafana/dashboards

# Set the Magna dashboard as the home dashboard
ENV GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/magna.json

EXPOSE 3000
