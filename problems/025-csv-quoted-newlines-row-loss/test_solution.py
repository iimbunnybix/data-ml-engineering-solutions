import csv
import io
import unittest

from solution import (
    MalformedRecord,
    Product,
    RowCountMismatch,
    load_products,
    naive_parse,
    parse_products,
)

PRODUCTS = [
    Product("SKU-1", "Kettle", "Steel kettle", 2499),
    Product("SKU-2", "Mug", "Ceramic, dishwasher safe", 799),
    Product("SKU-3", "Teapot", "Holds 1.2 litres.\nLid included, \"classic\" design", 3150),
    Product("SKU-4", "Spoon", "Plain", 199),
]


def vendor_export(products, line_ending="\r\n"):
    """Write products the way the vendor does: RFC 4180 CSV with quoting."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=line_ending)
    writer.writerow(["sku", "name", "description", "price_cents"])
    for product in products:
        writer.writerow([product.sku, product.name, product.description, product.price_cents])
    return buffer.getvalue()


class ReproduceProblem(unittest.TestCase):
    def test_naive_loader_silently_drops_rows_with_commas_or_newlines(self):
        loaded = naive_parse(vendor_export(PRODUCTS))
        self.assertEqual([product.sku for product in loaded], ["SKU-1", "SKU-4"])

    def test_naive_loader_reports_no_error_for_the_lost_rows(self):
        # Half the catalogue is gone and the job still "succeeds".
        self.assertEqual(len(naive_parse(vendor_export(PRODUCTS))), 2)


class FixParsesQuotedFields(unittest.TestCase):
    def test_all_rows_survive_with_exact_field_values(self):
        self.assertEqual(parse_products(vendor_export(PRODUCTS)), PRODUCTS)

    def test_unix_line_endings_parse_the_same(self):
        self.assertEqual(parse_products(vendor_export(PRODUCTS, "\n")), PRODUCTS)

    def test_embedded_newline_is_kept_inside_the_description(self):
        teapot = parse_products(vendor_export(PRODUCTS))[2]
        self.assertIn("\n", teapot.description)
        self.assertIn('"classic"', teapot.description)


class FixFailsLoudly(unittest.TestCase):
    def test_record_with_extra_column_is_rejected_not_skipped(self):
        text = vendor_export(PRODUCTS[:1]) + "SKU-9,Bad,Row,100,extra\r\n"
        with self.assertRaises(MalformedRecord):
            parse_products(text)

    def test_renamed_column_in_header_is_rejected(self):
        text = vendor_export(PRODUCTS).replace("price_cents", "price", 1)
        with self.assertRaises(MalformedRecord):
            parse_products(text)

    def test_empty_file_is_rejected(self):
        with self.assertRaises(MalformedRecord):
            parse_products("")

    def test_count_mismatch_with_manifest_fails_the_run(self):
        truncated = vendor_export(PRODUCTS[:3])     # vendor upload cut short
        with self.assertRaises(RowCountMismatch):
            load_products(truncated, manifest_row_count=4)

    def test_matching_manifest_count_loads_everything(self):
        self.assertEqual(load_products(vendor_export(PRODUCTS), manifest_row_count=4), PRODUCTS)


if __name__ == "__main__":
    unittest.main()
