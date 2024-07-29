#!/bin/bash

# Container entrypoint

# If any parameter is passed to the entrypoint, execute it
if [ $# -gt 0 ]; then
  exec "$@"
fi

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
