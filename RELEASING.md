# Releasing

Artifacts go to Maven Central through the [Sonatype Central Portal](https://central.sonatype.com), built and
uploaded by the GitHub Actions workflow `.github/workflows/release.yml`, the same way as dukehoff-mcp.
Published: `dukehoff-decide-parent`, `dukehoff-decide-core`, `dukehoff-decide-openai`,
`dukehoff-decide-spring-boot-starter`. Not published: `dukehoff-decide-showcase`.

## Setup

1. **Namespace.** `com.dukehoff` is verified in the Central Portal (DNS TXT record on `dukehoff.com`);
   `com.dukehoff.decide` falls under it.
2. **Signing key.** Releases are signed with the Dukehoff GmbH release key
   `323E745250E3722363642D1454583F1C96B286C0` ("Dukehoff GmbH Releases <opensource@dukehoff.com>", on
   keyserver.ubuntu.com, expires 2029-10-21). The `release` profile names it (`maven-gpg-plugin` `<keyname>`).
3. **Repository secrets** (Settings > Secrets and variables > Actions), as for dukehoff-mcp:
   ```
   gh secret set CENTRAL_USERNAME -R dukehoff-gmbh/dukehoff-decide        # Central Portal user token
   gh secret set CENTRAL_PASSWORD -R dukehoff-gmbh/dukehoff-decide
   gpg --export-secret-keys --armor 323E745250E3722363642D1454583F1C96B286C0 \
       | gh secret set MAVEN_GPG_PRIVATE_KEY -R dukehoff-gmbh/dukehoff-decide
   gh secret set MAVEN_GPG_PASSPHRASE -R dukehoff-gmbh/dukehoff-decide     # passphrase of the release key
   ```
   `gh secret set` asks for the value without echoing it. A Central token belongs to the account and works for
   every namespace in it.
4. **Optional variable** `CENTRAL_AUTO_PUBLISH=true` publishes uploads without the manual "Publish" step.

## Dry run

Actions > Release > Run workflow, with the version to try (e.g. `0.1.0`). It sets that version on the runner,
builds and tests, builds the Central bundle (signed if the key secrets are set) and checks it: only the four
published artifacts, every file inside its version folder, every jar and pom signed. Nothing is uploaded; the
bundle is kept as a workflow artifact for 7 days.

## Release

```
git tag -s v0.1.0 -m "dukehoff-decide 0.1.0"
git push origin v0.1.0
```

The tag starts the workflow. The version comes from the tag; the poms on `main` stay at `-SNAPSHOT` and nothing
is committed. The workflow fails right away if a secret is missing. After the same build, tests and bundle check
as the dry run, it uploads one deployment to the Central Portal and then creates the GitHub release.

The deployment is validated but **not published** (unless `CENTRAL_AUTO_PUBLISH` is true): check it under
"Deployments" in the portal and press "Publish". It usually appears on repo1.maven.org within about half an hour.
If validation fails, the plugin does not log why; the reasons are shown in the portal. Drop the failed deployment
there, fix the cause, delete the tag (`git push origin :v0.1.0`, `git tag -d v0.1.0`) and tag again.

Raise the `-SNAPSHOT` version on `main` when the next release is going to be a new minor or major version.

## Release from a laptop (fallback)

Needs a Central token in `~/.m2/settings.xml` (server `central`) and the release key unlocked:

```
echo test | gpg --local-user 323E745250E3722363642D1454583F1C96B286C0 --clearsign > /dev/null
mvn versions:set -DnewVersion=0.1.0 -DgenerateBackupPoms=false -DprocessAllModules=true
mvn -Prelease clean deploy -DskipUpload=true          # build and check target/central-publishing/central-bundle.zip
mvn -Prelease deploy                                  # upload
git checkout -- .                                     # back to -SNAPSHOT
```

The property is `skipUpload` (`skipPublishing` skips the bundle as well). `central-publishing-maven-plugin`
must be 0.12.0 or newer: older versions leave `maven-metadata-local.xml` next to the version folders under
Maven 3.10, and Central rejects such a bundle.
