/*
 * Copyright 2026 Dukehoff GmbH
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dukehoff.decide.showcase;

import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;

/**
 * The typed decision for one support ticket. Enum constants mirror the labels of
 * {@code Tobi-Bueck/customer-support-tickets}.
 */
@Question("You triage incoming customer support tickets for a software and IT services company. "
        + "Tickets may be in English or German.")
public record Triage(
        @Question("What kind of ticket is this?") Type type,
        @Question("Which queue should handle this ticket?") Queue queue,
        @Question("How urgent is this ticket?") Priority priority) {

    public enum Type {
        @Option("Something stopped working or is degraded right now") INCIDENT,
        @Option("Asks for information, access, a product or a service") REQUEST,
        @Option("A recurring or underlying issue that needs root-cause analysis") PROBLEM,
        @Option("Asks to modify, upgrade or reconfigure something") CHANGE
    }

    public enum Queue {
        @Option("Software bugs, errors and how-to questions about technical products") TECHNICAL_SUPPORT,
        @Option("Questions and issues about a specific product and its features") PRODUCT_SUPPORT,
        @Option("Accounts, orders, complaints and general customer concerns") CUSTOMER_SERVICE,
        @Option("Internal IT: hardware, networks, devices and access") IT_SUPPORT,
        @Option("Invoices, payments, refunds and subscriptions") BILLING_AND_PAYMENTS,
        @Option("Returning or exchanging products") RETURNS_AND_EXCHANGES,
        @Option("Outages, downtime and scheduled maintenance") SERVICE_OUTAGES_AND_MAINTENANCE,
        @Option("Pricing, quotes, demos and purchase inquiries") SALES_AND_PRE_SALES,
        @Option("Employment, payroll and HR matters") HUMAN_RESOURCES,
        @Option("Anything that fits no other queue") GENERAL_INQUIRY
    }

    public enum Priority {
        @Option("Business-critical, needs attention now") HIGH,
        @Option("Important, but work can continue") MEDIUM,
        @Option("Minor issue or general question") LOW
    }
}
