# Porting ES-document injection into ADT as native Java (`EsDataPrimer`)

Goal: ADT primes its own OpenSearch event data at test time with **zero `datagen.py`
shell-out**. Replace `StateProvider.runDatagenInject()` (and the `runDatagen(...)`
ephemeral-account path is out of scope; only ES priming is ported here) with an
in-process Java injector that mirrors `datamanager.py setup_data()`.

Verdict: **feasible and low-risk.** The injection internals are small (purge by term
query, template-fill, timestamp/randomize, bulk index into a date-partitioned index).
ADT already ships every library needed. No new heavy dependency required.

---

## 1. Dependency check (result: nothing new needed)

From `adt/gradle/01-dependencies.gradle` + `adt/gradle.properties`, ADT already has on
the **compile** classpath:

| Need | Already present | Version |
|---|---|---|
| HTTP client | `org.apache.httpcomponents.client5:httpclient5` (declared `implementation`) | 5.5 |
| JSON (tree + bind) | `com.fasterxml.jackson.core:jackson-databind` (declared `implementation`) | 2.21.2 |
| JSON (alt) | `com.google.code.gson:gson`, `org.json:json`, `json-simple` | 2.8.5 / 20171018 |
| REST convenience | `io.rest-assured:rest-assured` | 3.0.5 |
| commons-lang3 | present | 3.1 |

`BEAPIRequestor` already talks HTTP via **HttpClient 5** (`org.apache.hc.client5.http.impl.classic.HttpClients`).
So the port is a **thin REST client** built on the same `httpclient5` + Jackson ADT
already uses. **Do NOT add `opensearch-java` / the RestHighLevelClient** — it is a heavy
transitive tree (and the RHLC is Elasticsearch-7 licensed / version-coupled), unnecessary
for two endpoints (`POST /_bulk`, `POST <pattern>/_delete_by_query`).

Recommended: reuse Jackson `ObjectMapper` (already used across the `ND*Helper` classes).

**AWS SigV4 caveat (the one real gap):** `java-common >= 26.x` **dropped AWS SDK v1
entirely** (per the gradle comment) and ADT declares no AWS SDK. On devvm the local
OpenSearch is **plain HTTP, no auth** (confirmed: `curl http://localhost:9200` returns the
cluster banner with no credentials). Cloud `ndsdev`/`ndsuat` OpenSearch is
`vpc-...es.amazonaws.com:443` and datagen signs with **AWS4Auth** (see
`datagen/libs/elasticsearch/elasticsearch.py`). See Risk R2 for how ADT reaches the cloud
cluster today (it does not — and does not need to).

---

## 2. Index / doc shape (confirmed against the live cluster)

### Index name resolution (mirrors `DataManager.get_eventindex`)
```
index = DefaultEventIndex + utcfromtimestamp(request_time_ms/1000).strftime("%Y%m%d")
```
`DefaultEventIndex = eventindex_test_nudetect_score_` in **all three** env `.ini`
(`devvm/ndsdev/ndsuat`), so the final index is e.g.
`eventindex_test_nudetect_score_20260525`. `MaxRetentionTime = 90` is a *retention/cleanup*
knob (not used by injection) — the Java port can ignore it for priming. The `_v2` suffix
seen on some local indices is historical/manual; datagen writes the bare `<prefix><date>`
form, so the port does the same.

### Auto-mapping is FREE (big de-risk — verified live)
There is a cluster index template `eventindex-7.1` with `index_patterns: ["*eventindex_*"]`.
I POSTed a doc to a brand-new `eventindex_test_nudetect_score_29991231` and the index was
created **with the full event mapping applied automatically** (dynamic_templates for
`bas.counter.*`, keyword sub-fields, etc.), then deleted it. => The Java port does **not**
need to create indices or ship mappings. A plain bulk index into `<prefix><date>` inherits
the correct mapping. (This is exactly why datagen never creates the score indices itself.)

### Document shape
The `_source` is the template JSON verbatim with dotted-key overrides applied. Confirmed a
real doc in `eventindex_test_nudetect_score_20260524_v2` matches `datagen/data/test_data.json`
(371 lines: `schemaVersion`, `request_time`, `nd_website_id`, `bas.{ip,score,score_band}`,
`wsid`, `fid`, `rid`, `account.{id,last_24h}`, `action_data.AuthenticationData.*`,
`device.*`, `doa.anchor.{paths,values}` parallel arrays, `nscript.*`, etc.). Randomized
fields in the live doc (`wsid=wsc-...`, `email=Jn8sKBZsDE@automatetest.nds.com`) confirm the
randomizer output shape.

### The bulk action (what to POST to `/_bulk`)
For each generated doc, two NDJSON lines:
```
{"index":{"_index":"eventindex_test_nudetect_score_20260525"}}
{ ...full _source... }
```
No explicit `_id` (ES auto-generates — which is *why* purge-first is mandatory for
idempotency). `_type` is not needed on OS 1.3 (`_doc` is implicit).

---

## 3. Minimal Java design (`EsDataPrimer`)

Package: `com.nudata.tests.util.state` (next to `StateProvider`), or
`com.nudata.tests.util.es`.

### 3.1 Thin REST client `OpenSearchRestClient`
```java
final class OpenSearchRestClient implements AutoCloseable {
    private final String baseUrl;              // e.g. "http://localhost:9200"
    private final CloseableHttpClient http;    // org.apache.hc.client5...HttpClients.createDefault()
    private final ObjectMapper mapper;         // com.fasterxml.jackson
    private final RequestSigner signer;        // null on devvm; SigV4 on cloud (see R2)

    /** delete_by_query {term:{nd_website_id:id}} over "<prefix>*", conflicts=proceed&refresh=true */
    void deleteByWebsite(String indexPattern, String ndWebsiteId) {
        String url = baseUrl + "/" + indexPattern
            + "/_delete_by_query?conflicts=proceed&refresh=true&ignore_unavailable=true";
        String body = mapper.writeValueAsString(Map.of(
            "query", Map.of("term", Map.of("nd_website_id", ndWebsiteId))));
        post(url, body, "application/json");   // fail-loud on non-2xx
    }

    /** POST /_bulk with NDJSON; refresh=true so the seed is visible to the next test immediately */
    BulkResult bulk(List<String> ndjsonLines) {
        String url = baseUrl + "/_bulk?refresh=true";
        String body = String.join("\n", ndjsonLines) + "\n";
        String resp = post(url, body, "application/x-ndjson");
        // parse {"errors":false|true,"items":[...]} — fail-loud if errors==true
    }
}
```
Chunk the bulk (datagen uses `DEFAULT_BULK_CHUNK_SIZE=500` with a small inter-chunk pause
for the single-node devvm heap). Match that: flush every ~500 docs. Total ADT volume is
tiny (≈10–75 docs per feature section), so one chunk is usually enough.

### 3.2 Template + override model (ported fixtures, no Python at runtime)
Mirror `EventIndex` + `DataManager.transform_data` in Java:

```java
final class EventTemplate {
    private final JsonNode base;               // parsed once from a test-resource JSON
    JsonNode fresh() { return base.deepCopy(); }   // == EventIndex.reset_data()
    static void updateValue(ObjectNode root, String dottedKey, Object value) {
        // walk a.b.c; last segment set; THROW if a segment is missing (parity with
        // EventIndex.update_value which raises "key does not exist"). Handles account.id etc.
    }
}
```

Ported fixtures live under **ADT test resources**, not read from the datagen tree:
```
adt/src/test/resources/es-fixtures/
  templates/            <- copies of datagen/data/*.json actually used
    test_data.json, custom_data_contracts.json, test_bio_data.json,
    country_location_only.json, bas6_base_data.json, udid_data.json,
    20191125_purchase_data.json
  features.json         <- the .ini sections expressed as JSON (see below)
```

The `.ini` sections become one JSON descriptor per feature (replacing
`config/_shared/<feature>/*.ini`). Each section carries: template file, Count,
time config, Randomize list, and the dotted-key overrides with their `int:`/`str:`/etc.
type tags decoded to real JSON types. Example (`custom_data_contracts.ini` →):
```json
{
  "custom_data_contracts": {
    "template": "custom_data_contracts.json",
    "count": 1,
    "randomize": ["wsid","fid","rid"],
    "overrides": {
      "account.id": "custom@data.contracts",
      "email_domain": "custom@data.contracts",
      "email": "custom@data.contracts"
    }
  }
}
```
Type-tag decode rules (from `datamanager.transform_data`): bare=string,
`int:10`→10, `float:`→double, `str:foo`→literal "foo", `int_array:`/`float_array:`/
`string_array:`→arrays. This is ~15 lines of Java.

### 3.3 `EsDataPrimer` (mirrors `DataManager`)
```java
public final class EsDataPrimer {
    private final OpenSearchRestClient es;
    private final String indexPrefix;          // "eventindex_test_nudetect_score_"
    private final Map<String,FeatureSpec> features;   // loaded from features.json + templates/

    /** == purge_website_events: delete_by_query term nd_website_id over "<prefix>*" */
    public void purge(String baseId) {
        es.deleteByWebsite(indexPrefix + "*", baseId);
    }

    /** inject the shape of one feature section, stamped onto baseId (== transform_data + bulk) */
    public void inject(String featureKey, String baseId) {
        FeatureSpec spec = features.get(featureKey);            // template + count + overrides
        List<String> ndjson = new ArrayList<>();
        long[] ts = timestamps(spec);                          // get_time_from_config/fixed
        for (long t : ts) {
            ObjectNode doc = (ObjectNode) spec.template.fresh();
            EventTemplate.updateValue(doc, "nd_website_id", baseId);
            spec.overrides.forEach((k,v) -> EventTemplate.updateValue(doc, k, v));
            EventTemplate.updateValue(doc, "request_time", t);
            randomize(spec.randomize, doc);                    // wsid/fid/rid/account/did/...
            String index = indexPrefix + utcYyyymmdd(t);
            ndjson.add("{\"index\":{\"_index\":\""+index+"\"}}");
            ndjson.add(mapper.writeValueAsString(doc));
        }
        es.bulk(ndjson);
    }

    /** the primeBaseSites bulk seed: purge target once, then inject each coexisting feature onto it */
    public void primeBase(String baseId, List<String> featureKeys) {
        purge(baseId);
        for (String f : featureKeys) inject(f, baseId);
    }
}
```

`timestamps`, `utcYyyymmdd`, and `randomize` are direct 1:1 ports of the Python
(`get_timestamps` / `get_fixed_timestamps` / `parse_time_config` and the `EventIndex.randomize_*`
methods). The randomizer must reproduce the prefixes in
`eventindex_constants.py` (`wsc-`, `fc-`, `rc-`, `wcs-`, `ud1-`, `df2-`) and the
`@automatetest.nds.com` email domain, plus the **DOA-anchor parallel-array update**
(`doa.anchor.paths` index → `doa.anchor.values[index]`) used by `randomize_account/did/ip`.

---

## 4. The exact StateProvider seam

Three call sites shell out today (all funnel through the private
`runDatagenInject(websitesCsv, injectOnto, profile, label)`), plus `primeBaseSitesStatic`
which builds the coexisting feature CSVs. Replace as follows.

### 4.1 `primeBaseSites` / `primeBaseSitesStatic`
Today it joins feature KEYS into a CSV and shells `datagen -i True -w <csv> --inject-onto <base>`.
The **feature lists are already literally in the Java** (`bas8Features`, `bas6Features`).
Re-point them at the primer:
```java
public static void primeBaseSitesStatic(String siteBas8, String siteBas6, String profile) {
    EsDataPrimer primer = EsDataPrimer.forProfile(profile);   // resolves host/auth per env
    primer.primeBase(siteBas8, List.of(
        "event_index_data_generation", "event_index_data_generation_bas8_utc",
        "C9351", "C17291", "C18586",
        "bas8_datagen_aggregate_analysis", "bas8_datagen_nscript"));
    primer.primeBase(siteBas6, List.of(
        "bas6_datagen", "event_index_data_generation_bas6_non_pdt"));
}
```
Note: those 9 CSV tokens are **feature keys** (e.g. `C9351`, `bas8_datagen_nscript`), not
site ids. In datagen they resolve feature→ini folder via `config/<profile>/websites.json`.
The Java port **skips that indirection entirely** — the feature key maps directly to a
`FeatureSpec` in `features.json`, which already contains the template + overrides. So the
port also removes the per-profile `websites.json` feature→id lookup for priming.

### 4.2 `reseedSiteData(sourceFeatureSiteId, targetBaseSiteId, profile)`
This is the **shape-addressable re-prime**: inject the shape of a *source feature* but
purge+stamp onto a *target base*. In the port:
```java
public void reseedSiteData(String sourceFeatureKey, String targetBaseSiteId, String profile) {
    EsDataPrimer primer = EsDataPrimer.forProfile(profile);
    String base = (targetBaseSiteId == null || targetBaseSiteId.isBlank())
        ? sourceFeatureKey : targetBaseSiteId;   // legacy: source==target
    primer.purge(base);
    primer.inject(sourceFeatureKey, base);       // template from source, stamped onto base
}
```
Callers (ListAnchorTest dfp1/dfp2, BiometricsTest, CSVExportNonLeapTest) pass a **feature
key** as the source — same string they pass today as `-w`. One behavior change: today
`-w` accepts a *site id* that datagen reverse-maps to a feature; the port takes the
feature key directly. Since the existing callers already pass feature keys
(`event_index_data_generation_dfp1`, `bas8_datagen_biometrics`, ...), this is a no-op for
them; verify each call site's argument during implementation.

### 4.3 Delete the shell-out plumbing
`runDatagenInject(...)` (the ES one) is removed. Keep `runDatagen(...)` for the ephemeral
**account** provision path (that is a MySQL INSERT, not ES — out of scope for this port).
`ADT_DATAGEN_CMD` is no longer read for ES priming; add `ADT_ES_HOST` /
`ADT_ES_INDEX_PREFIX` system-properties instead (defaults: `http://localhost:9200`,
`eventindex_test_nudetect_score_`).

### 4.4 Unit/compile lane stays green
The class doc already notes the `@BeforeSuite` hook treats datagen-unavailable as SKIP.
Preserve that: `EsDataPrimer` should fail-loud on a genuine ES error but the suite hook
catches "cluster unreachable" and SKIPs so the no-OpenSearch compile lane is unaffected.

---

## 5. Risks / unknowns

**R1 — Index date-partition resolution.** LOW. Confirmed: `<prefix><yyyymmdd>` from
`request_time` UTC, prefix identical across envs, `MaxRetentionTime` irrelevant to
injection. The auto-apply template (`*eventindex_*`) means fresh date indices get correct
mappings with no work. The only subtlety: use **UTC** for the date (datagen uses
`datetime.utcfromtimestamp`), else docs near midnight land in the wrong index vs what the
dashboard queries. Port must use `Instant.ofEpochMilli(t).atZone(ZoneOffset.UTC)`.

**R2 — Cloud OpenSearch auth (the real unknown).** On devvm: plain HTTP, no auth — trivial.
On cloud `ndsuat`/`ndsdev` the OpenSearch domain is **VPC-internal**
(`vpc-...es.amazonaws.com:443`, AWS-SigV4). Two facts from the repo:
 (a) datagen signs with `AWS4Auth` (boto3 creds) — but that runs **on the in-VPC Jenkins
     node** (`envNdsuat`/`envNdsdev`), the only place the VPC ES is reachable. From devvm
     the cloud ES is NOT reachable at all (per memory [id=30087]).
 (b) ADT's `BEAPIRequestor` does **NOT** touch ES — it hits the dashboard BEAPI over HTTPS
     with HMAC/RSA (masterswitch `apiClient.profile.<env>`), no AWS signing anywhere in
     ADT, and `java-common>=26` dropped the AWS SDK.
 => Implication: an ADT run on the in-VPC agent CAN reach cloud ES, but **ADT has no SigV4
    signer today**. Options, in order of preference:
    - **(preferred) Keep ES priming a devvm/in-node-localhost concern.** The official
      pipeline already runs es-datagen (the provisioner) as a separate Jenkins stage before
      ADT; the Java port fully replaces the *devvm/local* `-i True` shell-out (the common
      case in these StateProvider runtime re-primes) and on cloud the base seeding stays
      with the existing datagen provision stage. This delivers "zero datagen.py at test
      time" for the local/dev loop without needing SigV4 in ADT.
    - **(if cloud runtime re-prime is required) add a minimal SigV4 signer.** Since the AWS
      SDK was intentionally dropped, implement a ~120-line HMAC-SHA256 SigV4 signer against
      `es` service (no SDK dep), fed by the agent's instance-role creds (env/instance
      metadata). This is the only way `reseedSiteData` can hit cloud ES in-process. Flag as
      a decision for the owner: it re-introduces AWS-signing surface ADT deliberately shed.
    - Resolve host/auth per profile in `EsDataPrimer.forProfile`: devvm→`http://localhost:9200`
      no signer; cloud→`https://<domain>:443` + SigV4 signer. Domain can be read from the
      same `datagen/config/<env>.ini [elasticsearch] Domain`, or vendored into ADT config.

**R3 — Mapping / refresh timing.** LOW-MED. datagen's `delete_by_query` uses
`refresh=true` and bulk uses default refresh; tests read immediately after priming. The
port must set `refresh=true` on BOTH the delete_by_query and the bulk (shown above) so the
seed is visible to the very next Selenium assertion. Without it, a fast test can query
before the segment is searchable (intermittent empty-result flake). datagen gets away with
laxer refresh because a full provision stage precedes ADT by minutes; runtime re-prime has
no such gap, so `refresh=true` on bulk is **required**, not optional.

**R4 — Randomizer fidelity.** LOW. The randomized fields (`wsid/fid/rid/did/account/ip`
+ DOA anchors) must match the Python prefixes and the parallel-array DOA update, or shape
assertions (ListAnchorTest DFP1/DFP2, endpoint_id composition `<ip>.<dfp1>`) break. This is
mechanical but must be ported carefully; unit-test the randomizer against a captured real
doc. `generate_random()` length/alphabet and `Randomizer.random_dfp()` need matching.

**R5 — Fixture drift.** MED (process, not technical). The ported `features.json` +
templates are a **copy** of `datagen/config/_shared/*` and `datagen/data/*`. If datagen's
fixtures change (as they did throughout PDD-2294), ADT's copy silently diverges. Mitigate
with a build-time check (checksum the vendored templates against the datagen tree when both
are present) or a documented "these are vendored from es-datagen @<commit>" header. Only
the **7 templates + ~11 feature sections ADT actually primes** need vendoring, not the
whole datagen config.

**R6 — `update_value` strictness.** LOW. Python `EventIndex.update_value` RAISES if a
dotted key path is missing (it does not create it). The Java port must do the same
(fail-loud) so a typo'd override is caught, not silently dropped. Note the known Python
limitation (memory [id=30087]): `update_value` cannot target list indices like
`doa.anchor.values.19` — the port inherits this and uses the dedicated DOA-anchor
path→value mechanism instead (already in `randomize_*`).

**R7 — Volume / heap.** LOW. ADT primes tens of docs per feature; even the full
`primeBase` union is a few hundred docs. The datagen chunking/back-pressure (500/chunk,
50ms pause) exists for the whole-cluster provision; ADT's runtime re-prime is small enough
that a single bulk call is fine. Keep a 500-doc chunk cap for safety.

---

## 6. Bottom line

- **Client:** thin REST on ADT's existing `httpclient5` (5.5) + Jackson (2.21.2). No new
  dep. Two endpoints: `POST <prefix>*/_delete_by_query?conflicts=proceed&refresh=true` and
  `POST /_bulk?refresh=true`.
- **Doc shape:** template JSON (vendored from `datagen/data/*.json`) + dotted-key overrides
  (vendored from the `.ini` sections as `features.json`) + ported randomizer; index =
  `eventindex_test_nudetect_score_<UTC-yyyymmdd(request_time)>`; auto-mapping via the
  existing `*eventindex_*` cluster template (no index creation needed — verified live).
- **Seam:** `EsDataPrimer.primeBase()` replaces `primeBaseSitesStatic`'s CSV shell-out;
  `primer.purge()+inject()` replaces `reseedSiteData`'s `runDatagenInject`. Feature KEYS
  map directly to `FeatureSpec` (drops the `websites.json` feature→id indirection). The ES
  `runDatagenInject` is deleted; the account-provision `runDatagen` stays.
- **Only real risk:** cloud SigV4 (R2) — ADT has no AWS signer and java-common dropped the
  SDK. Preferred resolution: the Java port owns the **local/dev** priming loop (zero
  datagen.py there), cloud base-seeding stays with the existing pre-ADT datagen provision
  stage; add a dependency-free SigV4 signer only if in-process cloud runtime re-prime is a
  hard requirement.
