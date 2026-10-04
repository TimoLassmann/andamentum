# API key rotation runbook notes

SaltLog service API keys must be rotated every 90 days.
Rotation uses the internal admin console: open Keys, choose the service, and select New key.
The old key stays valid for 24 hours after a new key is issued so services can switch over.
Each service reads its key from its secrets file, and the service must be restarted after the file changes.
After the switch, check the Usage page: the old key should show zero calls for one hour before it is revoked.
If a service still calls with the old key, do not revoke; find the caller from the Usage page first.
Record the rotation date in the team rotation log.
