#!/usr/bin/env python

"""
Convert CSV file to libsvm format. Works only with numeric variables.
Put -1 as label index (argv[3]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[4] == 1.

"""

import csv

import click


def construct_line(label, line):
    new_line = []
    if float(label) == 0.0:
        label = "0"
    new_line.append(label)

    for i, item in enumerate(line):
        if item == '' or float(item) == 0.0:
            continue
        new_item = "%s:%s" % (i + 1, item)
        new_line.append(new_item)
    new_line = " ".join(new_line)
    new_line += "\n"
    return new_line


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(exists=False), nargs=1)
@click.option('--label_index', type=int, default=-1)
@click.option('--no_label', is_flag=True)
@click.option('--skip_headers', is_flag=True)
def main(source, output, label_index, no_label, skip_headers):
    click.echo(click.format_filename(source))
    click.echo(click.format_filename(output))
    input_file = source
    output_file = output

    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            reader = csv.reader(fin)
            if skip_headers:
                next(reader)
            for line in reader:
                if no_label:
                    label = '1'
                else:
                    label = line.pop(label_index)

                new_line = construct_line(label, line)
                fout.write(new_line)


if __name__ == "__main__":
    main()
