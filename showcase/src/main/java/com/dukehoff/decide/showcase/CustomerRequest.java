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

/** The typed decision for one message of the Bitext customer support dataset. */
@Question("You route messages that customers send to an online shop's support chat.")
public record CustomerRequest(
        @Question("What is the message about?") Category category,
        @Question("Does the customer use offensive language, such as swearing or insults?") boolean offensive) {

    public enum Category {
        @Option("Creating, editing, deleting or switching an account, passwords, registration") ACCOUNT,
        @Option("Cancellation fees") CANCEL,
        @Option("Only asks how to reach customer service or a human agent, without naming any other topic "
                + "from this list") CONTACT,
        @Option("Delivery options and how long delivery takes") DELIVERY,
        @Option("Complaints and reviews") FEEDBACK,
        @Option("Finding or checking invoices") INVOICE,
        @Option("Placing, changing, cancelling or tracking an order") ORDER,
        @Option("Payment methods and payment problems") PAYMENT,
        @Option("Refund policy, getting or tracking a refund") REFUND,
        @Option("Setting up or changing a shipping address") SHIPPING,
        @Option("Newsletter subscription") SUBSCRIPTION
    }
}
