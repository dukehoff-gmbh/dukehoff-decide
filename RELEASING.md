# Releasing

Artifacts go to Maven Central through the [Sonatype Central Portal](https://central.sonatype.com).
Published: `dukehoff-decide-parent`, `dukehoff-decide-core`, `dukehoff-decide-openai`, `dukehoff-decide-spring-boot-starter`.
Not published: `dukehoff-decide-showcase`.

## Setup

1. **Namespace.** `com.dukehoff` is verified in the Central Portal (DNS TXT record on `dukehoff.com`);
   `com.dukehoff.decide` falls under it.
2. **Token.** Generate a user token in the portal (a separate one per machine, e.g. "laptop releases") and put it
   into `~/.m2/settings.xml`. A token belongs to the account and works for every namespace in it.
   ```xml
   <servers>
     <server>
       <id>central</id>
       <username>TOKEN_USERNAME</username>
       <password>TOKEN_PASSWORD</password>
     </server>
   </servers>
   ```
3. **Signing key.** Releases are signed with the Dukehoff GmbH release key
   `323E745250E3722363642D1454583F1C96B286C0` ("Dukehoff GmbH Releases <opensource@dukehoff.com>", on
   keyserver.ubuntu.com, expires 2029-10-21). The `release` profile names it (`maven-gpg-plugin` `<keyname>`), so
   it needs to be in the local keyring; nothing about signing goes into `settings.xml`. Unlock it before a release:
   ```
   echo test | gpg --local-user 323E745250E3722363642D1454583F1C96B286C0 --clearsign > /dev/null
   ```
4. **Project URL.** `github.url` in the parent `pom.xml` becomes `url` and `scm` of every published POM.

## Check first

```
mvn -Prelease clean verify
gpg --verify core/target/dukehoff-decide-core-*.jar.asc     # must say "Dukehoff GmbH Releases"
```

To also build the upload bundle without uploading it, use `mvn -Prelease clean deploy -DskipUpload=true` (the
property is `skipUpload`; `skipPublishing` skips the bundle as well). Every entry in the bundle must lie in a
version folder. `central-publishing-maven-plugin` must be 0.12.0 or newer: older versions leave
`maven-metadata-local.xml` next to the version folders under Maven 3.10, and Central rejects such a bundle.
To check the build without signing: `mvn -Prelease -Dgpg.skip verify`.

## Release

```
mvn versions:set -DnewVersion=0.1.0 -DgenerateBackupPoms=false
git commit -am "Release 0.1.0" && git tag -s v0.1.0 -m "dukehoff-decide 0.1.0"

mvn -Prelease clean deploy
```

The `release` profile adds sources and javadoc jars, signs everything and uploads one deployment. It is
validated but **not published** automatically (`autoPublish=false`): check it under "Deployments" in the
portal and press "Publish". It usually appears on repo1.maven.org within about half an hour.

If validation fails, the plugin does not log why; the reasons are shown in the portal under "Deployments".
Drop the failed deployment there before uploading again.

Then:

```
mvn versions:set -DnewVersion=0.2.0-SNAPSHOT -DgenerateBackupPoms=false
git commit -am "Start 0.2.0" && git push && git push --tags
```
